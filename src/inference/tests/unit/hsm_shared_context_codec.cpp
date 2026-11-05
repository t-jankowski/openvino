// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_shared_context_codec.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Encoder and decoder are checked against hand-written bytes on purpose, never against each other, so a mistake
// made symmetrically on both sides can't hide. A round trip belongs to the functional tests.
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

using Bytes = std::vector<uint8_t>;

static_assert(static_cast<uint8_t>(ov::element::Type_t::f32) == 4, "wire value of f32 changed");
static_assert(static_cast<uint8_t>(ov::element::Type_t::f8e8m0) == 25, "last valid element type changed");

static_assert(std::is_constructible_v<hsm::SharedContextEncoder, const hsm::SharedContextPayload&>);
static_assert(!std::is_constructible_v<hsm::SharedContextEncoder, hsm::SharedContextPayload>,
              "the encoder borrows its payload - a temporary would dangle");
static_assert(std::is_convertible_v<decltype(static_cast<std::optional<hsm::SharedContextPayload> (*)(
                                                 const hsm::Section&) noexcept>(&hsm::decode_shared_context)),
                                    hsm::SectionDecoder<hsm::SharedContextPayload>>,
              "decode_shared_context must be usable as a SectionDecoder");

// source: source_id 0x0102, offset 0x1000
const Bytes source_record{0x01, 0x10, 0x00, 0x00, 0x00,                     // kind, body_size = 16
                          0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,   // source_id
                          0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};  // offset

// constant: source_id 1, constant_id 2, offset 3, size 4, f32
const Bytes constant_record{0x02, 0x21, 0x00, 0x00, 0x00,                    // kind, body_size = 33
                            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // source_id
                            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // constant_id
                            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // offset
                            0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // size
                            0x04};                                           // element type f32

// model: model_id 5, offset 0x2000
const Bytes model_record{0x03, 0x10, 0x00, 0x00, 0x00,                     // kind, body_size = 16
                         0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,   // model_id
                         0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};  // offset

Bytes concat(std::initializer_list<Bytes> parts) {
    Bytes all;
    for (const auto& part : parts) {
        const auto at = all.size();
        all.resize(at + part.size());
        std::memcpy(all.data() + at, part.data(), part.size());
    }
    return all;
}

ov::util::MemoryView view_of(const Bytes& bytes) {
    return {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()};
}

Bytes encode(const hsm::SharedContextPayload& payload) {
    Bytes out;
    const hsm::SharedContextEncoder encoder{payload};
    encoder.encode([&out](ov::util::MemoryView chunk) {
        const auto at = out.size();
        out.resize(at + chunk.size());
        if (!chunk.empty()) {
            std::memcpy(out.data() + at, chunk.data(), chunk.size());
        }
    });
    return out;
}

ov::wsh::WeightMetaData meta(size_t offset, size_t size, ov::element::Type type) {
    return {offset, size, type};
}

hsm::Section pointer_section(const Bytes& bytes) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::shared_context_tag;
    entry.offset = hsm::multi_blob_alignment;
    entry.size = bytes.size();
    return hsm::Section{entry, view_of(bytes)};
}

}  // namespace

// --- Encoder ---

TEST(HsmSharedContextEncoderTest, empty_payload_writes_nothing) {
    const hsm::SharedContextPayload payload;
    EXPECT_TRUE(encode(payload).empty());
    EXPECT_EQ(hsm::SharedContextEncoder{payload}.size(), 0u);
}

TEST(HsmSharedContextEncoderTest, source_record_bytes) {
    hsm::SharedContextPayload payload;
    payload.sources[0x0102] = 0x1000;
    EXPECT_EQ(encode(payload), source_record);
}

TEST(HsmSharedContextEncoderTest, constant_record_bytes) {
    hsm::SharedContextPayload payload;
    payload.constants[1][2] = meta(3, 4, ov::element::f32);
    EXPECT_EQ(encode(payload), constant_record);
}

TEST(HsmSharedContextEncoderTest, model_record_bytes) {
    hsm::SharedContextPayload payload;
    payload.models[5] = 0x2000;
    EXPECT_EQ(encode(payload), model_record);
}

TEST(HsmSharedContextEncoderTest, records_are_ordered_by_kind_then_key_whatever_the_insertion_order) {
    hsm::SharedContextPayload payload;
    payload.models[5] = 0x2000;
    payload.constants[1][2] = meta(3, 4, ov::element::f32);
    payload.sources[0x0102] = 0x1000;
    payload.sources[0x0101] = 0x0800;

    const Bytes first_source{0x01, 0x10, 0x00, 0x00, 0x00,                     // kind, body_size
                             0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,   // source_id 0x0101
                             0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};  // offset 0x0800

    EXPECT_EQ(encode(payload), concat({first_source, source_record, constant_record, model_record}));
}

TEST(HsmSharedContextEncoderTest, constants_are_ordered_by_source_id_then_constant_id) {
    hsm::SharedContextPayload payload;
    payload.constants[2][1] = meta(0, 0, ov::element::u8);
    payload.constants[1][3] = meta(0, 0, ov::element::u8);
    payload.constants[1][2] = meta(3, 4, ov::element::f32);

    const auto out = encode(payload);
    ASSERT_EQ(out.size(), 3 * constant_record.size());

    // Each record is 38 bytes; its source_id starts at byte 5 and constant_id at byte 13.
    const auto source_of = [&out](size_t index) {
        return out[index * constant_record.size() + 5];
    };
    const auto constant_of = [&out](size_t index) {
        return out[index * constant_record.size() + 13];
    };
    EXPECT_EQ(source_of(0), 1);
    EXPECT_EQ(constant_of(0), 2);
    EXPECT_EQ(source_of(1), 1);
    EXPECT_EQ(constant_of(1), 3);
    EXPECT_EQ(source_of(2), 2);
    EXPECT_EQ(constant_of(2), 1);
}

TEST(HsmSharedContextEncoderTest, size_is_the_exact_encoded_length) {
    hsm::SharedContextPayload payload;
    payload.sources[1] = 1;
    payload.sources[2] = 2;
    payload.constants[1][1] = meta(0, 1, ov::element::i8);
    payload.constants[1][2] = meta(1, 1, ov::element::i8);
    payload.constants[2][1] = meta(0, 1, ov::element::i8);
    payload.models[9] = 3;

    constexpr size_t expected_size = 2 * 21 + 3 * 38 + 21;
    EXPECT_EQ(hsm::SharedContextEncoder{payload}.size(), expected_size);
    EXPECT_EQ(hsm::SharedContextEncoder{payload}.size(), encode(payload).size());
}

TEST(HsmSharedContextEncoderTest, reserved_external_sources_are_not_written) {
    hsm::SharedContextPayload payload;
    payload.external_sources[7] = {std::byte{1}, std::byte{2}};
    EXPECT_TRUE(encode(payload).empty());
    EXPECT_EQ(hsm::SharedContextEncoder{payload}.size(), 0u);
}

TEST(HsmSharedContextEncoderTest, output_does_not_depend_on_the_hash_table_layout) {
    hsm::SharedContextPayload ascending;
    hsm::SharedContextPayload descending;
    for (uint64_t id = 1; id <= 64; ++id) {
        ascending.sources[id] = id * 8;
        descending.sources[65 - id] = (65 - id) * 8;
    }
    EXPECT_EQ(encode(ascending), encode(descending));
}

// --- Decoder ---

TEST(HsmSharedContextDecoderTest, empty_bytes_are_an_empty_payload) {
    const auto payload = hsm::decode_shared_context(ov::util::MemoryView{});
    ASSERT_TRUE(payload.has_value());
    EXPECT_TRUE(payload->sources.empty());
    EXPECT_TRUE(payload->constants.empty());
    EXPECT_TRUE(payload->models.empty());
    EXPECT_TRUE(payload->external_sources.empty());
}

TEST(HsmSharedContextDecoderTest, reads_a_source_record) {
    const auto payload = hsm::decode_shared_context(view_of(source_record));
    ASSERT_TRUE(payload.has_value());
    ASSERT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->sources.at(0x0102), 0x1000u);
}

TEST(HsmSharedContextDecoderTest, reads_a_constant_record) {
    const auto payload = hsm::decode_shared_context(view_of(constant_record));
    ASSERT_TRUE(payload.has_value());
    ASSERT_EQ(payload->constants.size(), 1u);
    const auto& found = payload->constants.at(1).at(2);
    EXPECT_EQ(found.m_offset, 3u);
    EXPECT_EQ(found.m_size, 4u);
    EXPECT_EQ(found.m_type, ov::element::f32);
}

TEST(HsmSharedContextDecoderTest, reads_a_model_record) {
    const auto payload = hsm::decode_shared_context(view_of(model_record));
    ASSERT_TRUE(payload.has_value());
    ASSERT_EQ(payload->models.size(), 1u);
    EXPECT_EQ(payload->models.at(5), 0x2000u);
}

TEST(HsmSharedContextDecoderTest, reads_every_kind_in_one_payload_whatever_the_order) {
    const auto payload = hsm::decode_shared_context(view_of(concat({model_record, constant_record, source_record})));
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->constants.size(), 1u);
    EXPECT_EQ(payload->models.size(), 1u);
}

TEST(HsmSharedContextDecoderTest, the_same_constant_id_under_two_sources_is_not_a_duplicate) {
    Bytes other_source = constant_record;
    other_source[5] = 0x02;  // source_id 2

    const auto payload = hsm::decode_shared_context(view_of(concat({constant_record, other_source})));
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->constants.at(1).size(), 1u);
    EXPECT_EQ(payload->constants.at(2).size(), 1u);
}

TEST(HsmSharedContextDecoderTest, skips_a_record_of_an_unknown_kind) {
    const Bytes unknown{0x7F, 0x03, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC};
    const auto payload = hsm::decode_shared_context(view_of(concat({unknown, source_record})));
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
}

TEST(HsmSharedContextDecoderTest, skips_the_reserved_external_source_record_without_filling_anything) {
    const Bytes external{0x04, 0x02, 0x00, 0x00, 0x00, 0xAA, 0xBB};
    const auto payload = hsm::decode_shared_context(view_of(concat({external, model_record})));
    ASSERT_TRUE(payload.has_value());
    EXPECT_TRUE(payload->external_sources.empty());
    EXPECT_EQ(payload->models.size(), 1u);
}

TEST(HsmSharedContextDecoderTest, skips_an_unknown_record_with_an_empty_body) {
    const Bytes empty_body{0x7F, 0x00, 0x00, 0x00, 0x00};
    EXPECT_TRUE(hsm::decode_shared_context(view_of(empty_body)).has_value());
}

TEST(HsmSharedContextDecoderTest, only_a_cut_on_a_record_boundary_is_accepted) {
    const auto all = concat({source_record, constant_record, model_record});  // 21 + 38 + 21 bytes
    for (size_t length = 0; length <= all.size(); ++length) {
        const auto payload = hsm::decode_shared_context(ov::util::MemoryView{view_of(all).data(), length});
        const bool on_boundary = length == 0 || length == 21 || length == 59 || length == 80;
        EXPECT_EQ(payload.has_value(), on_boundary) << "length " << length;
    }
}

TEST(HsmSharedContextDecoderTest, rejects_a_body_longer_than_what_is_left) {
    Bytes lying = source_record;
    lying[1] = 0x11;  // body_size 17, only 16 follow
    EXPECT_FALSE(hsm::decode_shared_context(view_of(lying)).has_value());
}

TEST(HsmSharedContextDecoderTest, rejects_the_largest_possible_body_size_without_allocating_for_it) {
    const Bytes huge{0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x00};
    EXPECT_FALSE(hsm::decode_shared_context(view_of(huge)).has_value());
}

TEST(HsmSharedContextDecoderTest, rejects_a_known_record_with_a_body_of_the_wrong_size) {
    Bytes short_body = source_record;
    short_body[1] = 0x0F;  // 15 instead of 16
    short_body.pop_back();
    EXPECT_FALSE(hsm::decode_shared_context(view_of(short_body)).has_value());

    Bytes long_body = model_record;
    long_body[1] = 0x11;  // 17 instead of 16
    long_body.push_back(0x00);
    EXPECT_FALSE(hsm::decode_shared_context(view_of(long_body)).has_value());
}

TEST(HsmSharedContextDecoderTest, rejects_a_duplicate_key) {
    EXPECT_FALSE(hsm::decode_shared_context(view_of(concat({source_record, source_record}))).has_value());
    EXPECT_FALSE(hsm::decode_shared_context(view_of(concat({constant_record, constant_record}))).has_value());
    EXPECT_FALSE(hsm::decode_shared_context(view_of(concat({model_record, model_record}))).has_value());
}

TEST(HsmSharedContextDecoderTest, element_type_is_accepted_up_to_the_last_known_one_only) {
    Bytes last = constant_record;
    last.back() = 25;  // f8e8m0
    EXPECT_TRUE(hsm::decode_shared_context(view_of(last)).has_value());

    Bytes beyond = constant_record;
    beyond.back() = 26;
    EXPECT_FALSE(hsm::decode_shared_context(view_of(beyond)).has_value());

    Bytes garbage = constant_record;
    garbage.back() = 0xFF;
    EXPECT_FALSE(hsm::decode_shared_context(view_of(garbage)).has_value());
}

TEST(HsmSharedContextDecoderTest, rejects_a_constant_that_does_not_fit_size_t) {
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        Bytes wide = constant_record;
        wide[8 + 5 + 8 + 4] = 0x01;  // bit 32 of the offset
        EXPECT_FALSE(hsm::decode_shared_context(view_of(wide)).has_value());
    } else {
        GTEST_SKIP() << "every u64 fits a 64-bit size_t";
    }
}

// --- Decoding from a Section ---

TEST(HsmSharedContextDecoderTest, reads_a_buffer_backed_section) {
    const auto bytes = concat({source_record, model_record});
    const auto payload = hsm::decode_shared_context(pointer_section(bytes));
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->models.size(), 1u);
}

TEST(HsmSharedContextDecoderTest, reads_a_stream_backed_section) {
    const auto bytes = concat({source_record, model_record});
    std::istringstream stream{std::string{bytes.begin(), bytes.end()}};

    hsm::ManifestEntry entry{};
    entry.tag = hsm::shared_context_tag;
    entry.offset = 0;
    entry.size = bytes.size();
    const hsm::Section section{entry, stream, std::streampos(0)};

    const auto payload = hsm::decode_shared_context(section);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->models.size(), 1u);
}

TEST(HsmSharedContextDecoderTest, an_empty_section_is_an_empty_payload) {
    const auto payload = hsm::decode_shared_context(pointer_section({}));
    ASSERT_TRUE(payload.has_value());
    EXPECT_TRUE(payload->sources.empty());
}

TEST(HsmSharedContextDecoderTest, a_stream_that_ends_early_is_rejected_without_throwing) {
    std::istringstream stream{std::string(source_record.size() - 4, '\0')};

    hsm::ManifestEntry entry{};
    entry.tag = hsm::shared_context_tag;
    entry.size = source_record.size();
    const hsm::Section section{entry, stream, std::streampos(0)};

    EXPECT_FALSE(hsm::decode_shared_context(section).has_value());
}

TEST(HsmSharedContextDecoderTest, a_malformed_section_is_rejected) {
    const Bytes truncated{0x01, 0x10, 0x00};
    EXPECT_FALSE(hsm::decode_shared_context(pointer_section(truncated)).has_value());
}

}  // namespace ov::test
