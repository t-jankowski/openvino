// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_multi_blob_writer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_shared_context_codec.hpp"
#include "openvino/util/mmap_object.hpp"

// What MultiBlobWriter writes is read back with the real MultiBlobView, Reader and shared-context decoder.
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

constexpr uint64_t source_a = 5, source_b = 9, source_c = 2;
constexpr size_t size_a = 6000, size_b = 100, size_c = 8192;
constexpr uint64_t model_x = 100, model_y = 200;

ov::ICacheManager::StreamWriter export_model(const std::string& payload) {
    return [payload](std::ostream& stream) {
        auto writer = hsm::DeferredWriter::open(stream);
        ASSERT_TRUE(writer.has_value());
        ASSERT_TRUE(writer->add_section(
            hsm::any_device_id,
            hsm::model_tag,
            ov::util::MemoryView{reinterpret_cast<const std::byte*>(payload.data()), payload.size()}));
        ASSERT_EQ(writer->finalize(), std::error_code{});
    };
}

class Buffers {
public:
    void add(ov::wsh::Context& context, ov::wsh::DataID id, size_t size, uint8_t fill) {
        auto buffer = std::make_shared<ov::AlignedBuffer>(size);
        std::memset(buffer->get_ptr(), fill, size);
        context.m_cache_sources[id] = ov::wsh::WeightSource{"", buffer};
        m_keep.push_back(std::move(buffer));
    }

private:
    std::vector<std::shared_ptr<ov::AlignedBuffer>> m_keep;
};

struct SharedContainer {
    uint64_t offset;
    uint64_t size;
};

// Start and size of every shared-context container, found by stepping over the file the way a reader does.
std::vector<SharedContainer> shared_containers(const std::byte* data, size_t size) {
    std::vector<SharedContainer> found;
    for (uint64_t offset = 0; offset < size;) {
        const auto& header = hsm::Header::view(static_cast<const void*>(data + offset));
        if (header.magic == hsm::BlobMagic::multi) {
            found.push_back({offset, header.container_size});
        }
        offset += header.container_size;
    }
    return found;
}

bool all_bytes_are(const std::byte* data, uint64_t offset, size_t size, uint8_t expected) {
    return std::all_of(data + offset, data + offset + size, [expected](std::byte value) {
        return static_cast<uint8_t>(value) == expected;
    });
}

class HsmMultiBlobRoundTripTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_path = ov::test::utils::generateTestFilePrefix() + ".bin";
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::remove(m_path, ignored);
    }

    std::filesystem::path m_path;
};

}  // namespace

TEST_F(HsmMultiBlobRoundTripTest, a_written_file_reads_back_through_the_view_the_reader_and_the_decoder) {
    Buffers buffers;
    {
        hsm::MultiBlobWriter writer{m_path};

        ov::wsh::Context first;
        buffers.add(first, source_a, size_a, 0x5A);
        buffers.add(first, source_b, size_b, 0x09);
        first.m_weight_registry[source_a][1] = {0, 4000, ov::element::f16};
        first.m_weight_registry[source_a][2] = {4000, 2000, ov::element::i8};
        writer.write_context(first);

        writer.write_model(model_x, export_model("model-100"));
        writer.write_model(model_y, export_model("model-200"));

        ov::wsh::Context second = first;
        buffers.add(second, source_c, size_c, 0x02);
        second.m_weight_registry[source_c][1] = {0, size_c, ov::element::u8};
        writer.write_context(second);
        writer.close();
    }

    const auto mapped = ov::load_mmap_object(m_path);
    const std::byte* data = mapped->data();

    // The view finds the model blobs and skips every shared context.
    const hsm::MultiBlobView view{data, mapped->size()};
    ASSERT_EQ(view.blob_count(), 2u);
    for (size_t i = 0; i < view.blob_count(); ++i) {
        EXPECT_TRUE(view.blob_at(i).validate());
    }

    // Each model blob opens as a container of its own and still holds what the plugin wrote.
    const auto model_payload = [&](size_t index) {
        const auto blob = view.blob_at(index);
        const auto reader = hsm::Reader::open(reinterpret_cast<const std::byte*>(&blob.header()), blob.size());
        EXPECT_TRUE(reader.has_value());
        const auto section = reader->section(hsm::any_device_id, hsm::model);
        EXPECT_TRUE(section.has_value());
        const auto bytes = section->view();
        return std::string{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    };
    EXPECT_EQ(model_payload(0), "model-100");
    EXPECT_EQ(model_payload(1), "model-200");

    // The shared contexts, merged, describe the whole file.
    hsm::SharedContextPayload merged;
    const auto containers = shared_containers(data, mapped->size());
    ASSERT_EQ(containers.size(), 3u);  // the empty first one, then the two deltas
    for (const auto& container : containers) {
        const auto reader = hsm::Reader::open(data + container.offset, container.size);
        ASSERT_TRUE(reader.has_value());
        const auto section = reader->section(hsm::any_device_id, hsm::shared_context);
        if (!section) {
            continue;  // the empty first container carries no payload
        }
        const auto payload = hsm::decode_shared_context(*section);
        ASSERT_TRUE(payload.has_value());
        merged.sources.insert(payload->sources.begin(), payload->sources.end());
        merged.models.insert(payload->models.begin(), payload->models.end());
        for (const auto& [source, metas] : payload->constants) {
            merged.constants[source].insert(metas.begin(), metas.end());
        }
    }

    // Every source link leads to the bytes of that source, on a page boundary.
    ASSERT_EQ(merged.sources.size(), 3u);
    for (const auto& [id, link] : merged.sources) {
        EXPECT_EQ(link % hsm::multi_blob_alignment, 0u) << "source " << id;
    }
    EXPECT_TRUE(all_bytes_are(data, merged.sources.at(source_a), size_a, 0x5A));
    EXPECT_TRUE(all_bytes_are(data, merged.sources.at(source_b), size_b, 0x09));
    EXPECT_TRUE(all_bytes_are(data, merged.sources.at(source_c), size_c, 0x02));

    // The constants are all there with their places in their sources.
    ASSERT_EQ(merged.constants.size(), 2u);
    EXPECT_EQ(merged.constants.at(source_a).at(1).m_size, 4000u);
    EXPECT_EQ(merged.constants.at(source_a).at(1).m_type, ov::element::f16);
    EXPECT_EQ(merged.constants.at(source_a).at(2).m_offset, 4000u);
    EXPECT_EQ(merged.constants.at(source_c).at(1).m_type, ov::element::u8);

    // The model index points exactly at the blobs the view finds, in the order they were written.
    ASSERT_EQ(merged.models.size(), 2u);
    EXPECT_EQ(data + merged.models.at(model_x), reinterpret_cast<const std::byte*>(&view.blob_at(0).header()));
    EXPECT_EQ(data + merged.models.at(model_y), reinterpret_cast<const std::byte*>(&view.blob_at(1).header()));
}

TEST_F(HsmMultiBlobRoundTripTest, a_shared_context_can_be_read_through_a_stream_too) {
    Buffers buffers;
    ov::wsh::Context context;
    buffers.add(context, source_a, size_a, 0x5A);
    context.m_weight_registry[source_a][1] = {0, size_a, ov::element::u8};
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_context(context);
    }

    const auto mapped = ov::load_mmap_object(m_path);
    const auto containers = shared_containers(mapped->data(), mapped->size());
    ASSERT_EQ(containers.size(), 2u);

    std::ifstream stream{m_path, std::ios::binary};
    stream.seekg(static_cast<std::streamoff>(containers[1].offset));
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::any_device_id, hsm::shared_context);
    ASSERT_TRUE(section.has_value());
    EXPECT_TRUE(section->view().empty());  // stream-backed: no zero-copy view

    const auto payload = hsm::decode_shared_context(*section);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->constants.at(source_a).at(1).m_size, size_a);
}

TEST_F(HsmMultiBlobRoundTripTest, appending_to_a_reopened_file_keeps_it_readable) {
    Buffers buffers;
    ov::wsh::Context first;
    buffers.add(first, source_a, size_a, 0x5A);
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_context(first);
        writer.write_model(model_x, export_model("model-100"));
    }
    {
        hsm::MultiBlobWriter reopened{m_path};
        ov::wsh::Context second = first;
        buffers.add(second, source_b, size_b, 0x09);
        reopened.write_context(second);
        reopened.write_model(model_y, export_model("model-200"));
    }

    const auto mapped = ov::load_mmap_object(m_path);
    const hsm::MultiBlobView view{mapped->data(), mapped->size()};
    ASSERT_EQ(view.blob_count(), 2u);
    EXPECT_TRUE(view.blob_at(0).validate());
    EXPECT_TRUE(view.blob_at(1).validate());

    // Source A was written once, by the first writer; the second one only added B.
    size_t weights_sections = 0;
    for (const auto& container : shared_containers(mapped->data(), mapped->size())) {
        const auto reader = hsm::Reader::open(mapped->data() + container.offset, container.size);
        ASSERT_TRUE(reader.has_value());
        weights_sections += reader->count(hsm::any_device_id, hsm::weights);
    }
    EXPECT_EQ(weights_sections, 2u);
}

}  // namespace ov::test
