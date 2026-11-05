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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_shared_context_codec.hpp"

// The file is parsed by hand (header, manifest, offsets) instead of through Reader or MultiBlobView, so the writer
// is judged against the documented layout and not against code that might share its mistakes.
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

constexpr uint64_t alignment = 4096;

using Bytes = std::vector<uint8_t>;

Bytes read_file(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    return Bytes{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

void write_file(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

struct Container {
    uint64_t offset = 0;
    hsm::Header header{};
    std::vector<hsm::ManifestEntry> entries;
};

std::vector<Container> parse_file(const Bytes& bytes) {
    std::vector<Container> containers;
    uint64_t offset = 0;
    while (offset < bytes.size()) {
        if (bytes.size() - offset < sizeof(hsm::Header)) {
            ADD_FAILURE() << "cut-off header at " << offset;
            break;
        }
        Container container;
        container.offset = offset;
        std::memcpy(&container.header, bytes.data() + offset, sizeof(hsm::Header));
        const auto& header = container.header;
        if (!hsm::is_valid_header_fields(header) || header.container_size > bytes.size() - offset) {
            ADD_FAILURE() << "invalid container at " << offset;
            break;
        }
        container.entries.resize(header.manifest_size / sizeof(hsm::ManifestEntry));
        if (!container.entries.empty()) {
            std::memcpy(container.entries.data(), bytes.data() + offset + header.manifest_offset, header.manifest_size);
        }
        offset += header.container_size;
        containers.push_back(std::move(container));
    }
    return containers;
}

std::vector<hsm::ManifestEntry> entries_with(const Container& container, uint32_t tag_id) {
    std::vector<hsm::ManifestEntry> found;
    for (const auto& entry : container.entries) {
        if (entry.tag.id() == tag_id) {
            found.push_back(entry);
        }
    }
    return found;
}

std::optional<hsm::SharedContextPayload> payload_of(const Bytes& bytes, const Container& container) {
    const auto found = entries_with(container, hsm::shared_context);
    if (found.size() != 1) {
        return hsm::decode_shared_context(ov::util::MemoryView{});  // no section: an empty payload
    }
    return hsm::decode_shared_context(
        ov::util::MemoryView{reinterpret_cast<const std::byte*>(bytes.data() + container.offset + found.front().offset),
                             static_cast<size_t>(found.front().size)});
}

// Stands for a plugin exporting its compiled model.
ov::ICacheManager::StreamWriter model_blob(std::string payload = "model-bytes") {
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

// Owns the buffers a Context only observes.
class Sources {
public:
    void add(ov::wsh::DataID id, size_t size, uint8_t fill) {
        auto buffer = std::make_shared<ov::AlignedBuffer>(size);
        std::memset(buffer->get_ptr(), fill, size);
        context.m_cache_sources[id] = ov::wsh::WeightSource{"", buffer};
        m_buffers.push_back(std::move(buffer));
    }

    void add_constant(ov::wsh::DataID source, ov::wsh::DataID constant, size_t offset, size_t size) {
        context.m_weight_registry[source][constant] = ov::wsh::WeightMetaData{offset, size, ov::element::u8};
    }

    void drop(ov::wsh::DataID id) {
        m_buffers.erase(std::remove_if(m_buffers.begin(),
                                       m_buffers.end(),
                                       [&](const auto& buffer) {
                                           return context.m_cache_sources.at(id).m_weights.lock() == buffer;
                                       }),
                        m_buffers.end());
    }

    ov::wsh::Context context;

private:
    std::vector<std::shared_ptr<ov::AlignedBuffer>> m_buffers;
};

Bytes header_only_container(hsm::BlobMagic magic) {
    hsm::Header header{};
    header.magic = magic;
    header.version_major = hsm::FormatVersion::major;
    header.version_minor = hsm::FormatVersion::minor;
    header.container_size = sizeof(hsm::Header);
    header.manifest_offset = sizeof(hsm::Header);
    Bytes bytes(sizeof(header));
    std::memcpy(bytes.data(), &header, sizeof(header));
    return bytes;
}

}  // namespace

class HsmMultiBlobWriterTest : public ::testing::Test {
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

// --- Creating and opening ---

TEST_F(HsmMultiBlobWriterTest, a_new_file_starts_with_an_empty_shared_context) {
    { hsm::MultiBlobWriter writer{m_path}; }

    const auto bytes = read_file(m_path);
    ASSERT_EQ(bytes.size(), alignment);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 1u);
    EXPECT_EQ(containers[0].header.magic, hsm::BlobMagic::multi);
    EXPECT_EQ(containers[0].header.container_size, alignment);  // padded up to the boundary
    EXPECT_TRUE(containers[0].entries.empty());
}

TEST_F(HsmMultiBlobWriterTest, missing_parent_directories_are_created) {
    const auto directory = m_path.parent_path() / (m_path.stem().string() + "_dir");
    const auto nested = directory / "nested" / "file.bin";
    { hsm::MultiBlobWriter writer{nested}; }
    EXPECT_TRUE(std::filesystem::exists(nested));
    std::filesystem::remove_all(directory);
}

TEST_F(HsmMultiBlobWriterTest, an_empty_existing_file_is_treated_as_new) {
    write_file(m_path, {});
    { hsm::MultiBlobWriter writer{m_path}; }
    EXPECT_EQ(read_file(m_path).size(), alignment);
}

TEST_F(HsmMultiBlobWriterTest, reopening_a_valid_file_keeps_it_as_it_is) {
    { hsm::MultiBlobWriter writer{m_path}; }
    const auto before = read_file(m_path);
    { hsm::MultiBlobWriter writer{m_path}; }
    EXPECT_EQ(read_file(m_path), before);
}

// --- Shared context ---

TEST_F(HsmMultiBlobWriterTest, new_sources_are_written_as_aligned_weights_sections_that_the_links_point_at) {
    Sources sources;
    sources.add(3, 5000, 0xA3);
    sources.add(7, 10, 0xB7);
    sources.add_constant(3, 1, 0, 16);

    hsm::MultiBlobWriter writer{m_path};
    writer.write_context(sources.context);

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 2u);
    const auto& delta = containers[1];
    EXPECT_EQ(delta.offset, alignment);
    EXPECT_EQ(delta.header.magic, hsm::BlobMagic::multi);
    EXPECT_EQ(delta.header.container_size % alignment, 0u);

    const auto weights = entries_with(delta, hsm::weights);
    ASSERT_EQ(weights.size(), 2u);
    EXPECT_EQ(weights[0].offset, alignment);  // source 3 first: sources go in id order
    EXPECT_EQ(weights[0].size, 5000u);
    EXPECT_EQ(weights[1].offset, 3 * alignment);  // 4096 + 5000, rounded up
    EXPECT_EQ(weights[1].size, 10u);

    const auto payload = payload_of(bytes, delta);
    ASSERT_TRUE(payload.has_value());
    ASSERT_EQ(payload->sources.size(), 2u);
    EXPECT_EQ(payload->sources.at(3), delta.offset + alignment);      // 8192
    EXPECT_EQ(payload->sources.at(7), delta.offset + 3 * alignment);  // 16384
    EXPECT_EQ(bytes[payload->sources.at(3)], 0xA3);
    EXPECT_EQ(bytes[payload->sources.at(3) + 4999], 0xA3);
    EXPECT_EQ(bytes[payload->sources.at(7)], 0xB7);
    EXPECT_EQ(bytes[payload->sources.at(7) + 9], 0xB7);

    ASSERT_EQ(payload->constants.size(), 1u);
    EXPECT_EQ(payload->constants.at(3).at(1).m_size, 16u);
}

TEST_F(HsmMultiBlobWriterTest, a_delta_holds_only_what_the_file_does_not_know_yet) {
    Sources sources;
    sources.add(3, 100, 0x03);
    sources.add_constant(3, 1, 0, 8);

    hsm::MultiBlobWriter writer{m_path};
    writer.write_context(sources.context);

    sources.add(7, 100, 0x07);
    sources.add_constant(3, 2, 8, 8);
    sources.add_constant(7, 1, 0, 8);
    writer.write_context(sources.context);

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 3u);

    const auto second = payload_of(bytes, containers[2]);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(entries_with(containers[2], hsm::weights).size(), 1u);
    ASSERT_EQ(second->sources.size(), 1u);
    EXPECT_EQ(second->sources.count(7), 1u);
    EXPECT_EQ(second->constants.at(3).size(), 1u);  // only constant 2; constant 1 is already in the file
    EXPECT_EQ(second->constants.at(3).count(2), 1u);
    EXPECT_EQ(second->constants.at(7).count(1), 1u);
}

TEST_F(HsmMultiBlobWriterTest, nothing_is_appended_when_nothing_is_new) {
    Sources sources;
    sources.add(3, 100, 0x03);

    hsm::MultiBlobWriter writer{m_path};
    writer.write_context(sources.context);
    const auto size = std::filesystem::file_size(m_path);

    writer.write_context(sources.context);
    writer.write_context(ov::wsh::Context{});
    EXPECT_EQ(std::filesystem::file_size(m_path), size);
}

TEST_F(HsmMultiBlobWriterTest, constants_of_a_source_the_file_will_never_have_are_left_out) {
    Sources sources;
    sources.add_constant(42, 1, 0, 8);  // no buffer for source 42

    hsm::MultiBlobWriter writer{m_path};
    writer.write_context(sources.context);
    EXPECT_EQ(std::filesystem::file_size(m_path), alignment);
}

TEST_F(HsmMultiBlobWriterTest, a_source_whose_buffer_is_gone_is_not_written) {
    Sources sources;
    sources.add(3, 100, 0x03);
    sources.add(7, 100, 0x07);
    sources.drop(3);

    hsm::MultiBlobWriter writer{m_path};
    writer.write_context(sources.context);

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 2u);
    const auto payload = payload_of(bytes, containers[1]);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->sources.size(), 1u);
    EXPECT_EQ(payload->sources.count(7), 1u);
}

TEST_F(HsmMultiBlobWriterTest, a_reopened_file_is_not_given_sources_it_already_has) {
    Sources sources;
    sources.add(3, 100, 0x03);
    sources.add_constant(3, 1, 0, 8);
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_context(sources.context);
    }
    const auto size = std::filesystem::file_size(m_path);

    hsm::MultiBlobWriter reopened{m_path};
    reopened.write_context(sources.context);
    EXPECT_EQ(std::filesystem::file_size(m_path), size);
}

TEST_F(HsmMultiBlobWriterTest, external_sources_are_accepted_and_change_nothing) {
    hsm::MultiBlobWriter writer{m_path};
    const std::byte descriptor[3] = {};
    EXPECT_NO_THROW(writer.register_external_source(5, ov::util::MemoryView{descriptor, 3}));
    EXPECT_EQ(std::filesystem::file_size(m_path), alignment);
}

// --- Model blobs ---

TEST_F(HsmMultiBlobWriterTest, a_model_blob_starts_on_the_boundary_and_its_padding_counts_in_its_size) {
    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob());

    const auto containers = parse_file(read_file(m_path));
    ASSERT_EQ(containers.size(), 2u);
    EXPECT_EQ(containers[1].offset, alignment);
    EXPECT_EQ(containers[1].header.magic, hsm::BlobMagic::single);
    EXPECT_EQ(containers[1].header.container_size, alignment);
    EXPECT_EQ(std::filesystem::file_size(m_path), 2 * alignment);
}

TEST_F(HsmMultiBlobWriterTest, a_large_model_blob_is_padded_up_to_the_next_boundary) {
    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob(std::string(5000, 'x')));

    const auto containers = parse_file(read_file(m_path));
    ASSERT_EQ(containers.size(), 2u);
    EXPECT_EQ(containers[1].header.container_size, 2 * alignment);
}

TEST_F(HsmMultiBlobWriterTest, the_model_record_waits_for_close) {
    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob());
    EXPECT_EQ(parse_file(read_file(m_path)).size(), 2u);  // blob only, not indexed yet

    writer.close();

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 3u);
    EXPECT_EQ(containers[2].header.magic, hsm::BlobMagic::multi);
    const auto payload = payload_of(bytes, containers[2]);
    ASSERT_TRUE(payload.has_value());
    ASSERT_EQ(payload->models.size(), 1u);
    EXPECT_EQ(payload->models.at(11), alignment);  // where the blob starts
}

TEST_F(HsmMultiBlobWriterTest, the_model_record_goes_out_with_the_next_context) {
    Sources sources;
    sources.add(3, 100, 0x03);

    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob());
    writer.write_context(sources.context);

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 3u);
    const auto payload = payload_of(bytes, containers[2]);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->models.at(11), alignment);
    EXPECT_EQ(payload->sources.size(), 1u);
}

TEST_F(HsmMultiBlobWriterTest, a_model_record_alone_makes_a_context_worth_writing) {
    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob());
    writer.write_context(ov::wsh::Context{});
    EXPECT_EQ(parse_file(read_file(m_path)).size(), 3u);
}

TEST_F(HsmMultiBlobWriterTest, releasing_the_last_writer_writes_the_model_record) {
    { hsm::MultiBlobWriter(m_path).write_model(11, model_blob()); }

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 3u);
    const auto payload = payload_of(bytes, containers[2]);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->models.count(11), 1u);
}

TEST_F(HsmMultiBlobWriterTest, a_model_id_cannot_be_written_twice) {
    hsm::MultiBlobWriter writer{m_path};
    writer.write_model(11, model_blob());
    EXPECT_THROW(writer.write_model(11, model_blob()), ov::Exception);  // not indexed yet
    writer.close();
    const auto size = std::filesystem::file_size(m_path);

    EXPECT_THROW(writer.write_model(11, model_blob()), ov::Exception);  // indexed
    EXPECT_EQ(std::filesystem::file_size(m_path), size);
}

TEST_F(HsmMultiBlobWriterTest, a_reopened_file_still_refuses_an_indexed_model_id) {
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_model(11, model_blob());
    }
    hsm::MultiBlobWriter reopened{m_path};
    EXPECT_THROW(reopened.write_model(11, model_blob()), ov::Exception);
    EXPECT_NO_THROW(reopened.write_model(12, model_blob()));
}

TEST_F(HsmMultiBlobWriterTest, a_model_blob_without_an_index_record_is_skipped_when_scanning) {
    const auto copy = m_path.string() + ".copy";
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_model(11, model_blob());
        std::filesystem::copy_file(m_path,
                                   copy,
                                   std::filesystem::copy_options::overwrite_existing);  // as after a crash
    }

    hsm::MultiBlobWriter reopened{copy};
    EXPECT_NO_THROW(reopened.write_model(11, model_blob()));  // never indexed, so not known
    std::filesystem::remove(copy);
}

// --- Failures leave the file as it was ---

TEST_F(HsmMultiBlobWriterTest, a_failing_export_is_cut_back_and_the_id_stays_free) {
    hsm::MultiBlobWriter writer{m_path};
    const auto size = std::filesystem::file_size(m_path);

    EXPECT_THROW(writer.write_model(11,
                                    [](std::ostream& stream) {
                                        stream << std::string(6000, 'x');
                                        throw std::runtime_error{"export failed"};
                                    }),
                 std::runtime_error);
    EXPECT_EQ(std::filesystem::file_size(m_path), size);

    EXPECT_NO_THROW(writer.write_model(11, model_blob()));
}

TEST_F(HsmMultiBlobWriterTest, an_export_that_is_not_exactly_one_model_blob_is_refused) {
    hsm::MultiBlobWriter writer{m_path};
    const auto size = std::filesystem::file_size(m_path);

    const auto expect_refused = [&](const ov::ICacheManager::StreamWriter& export_model) {
        EXPECT_THROW(writer.write_model(11, export_model), ov::Exception);
        EXPECT_EQ(std::filesystem::file_size(m_path), size);
    };

    expect_refused([](std::ostream&) {});  // nothing
    expect_refused([](std::ostream& stream) {
        stream << std::string(100, 'x');  // not a container
    });
    expect_refused([](std::ostream& stream) {
        model_blob()(stream);  // two containers
        model_blob()(stream);
    });
    expect_refused([](std::ostream& stream) {
        auto shared_context = hsm::DeferredWriter::open(stream, hsm::BlobMagic::multi);  // the wrong kind
        ASSERT_TRUE(shared_context.has_value());
        ASSERT_EQ(shared_context->finalize(), std::error_code{});
    });
}

// --- Invalid files are never repaired ---

TEST_F(HsmMultiBlobWriterTest, an_invalid_file_is_refused_and_left_alone) {
    { hsm::MultiBlobWriter writer{m_path}; }
    const auto valid = read_file(m_path);

    const auto expect_refused = [&](const Bytes& content) {
        write_file(m_path, content);
        EXPECT_THROW(hsm::MultiBlobWriter{m_path}, ov::Exception);
        EXPECT_EQ(read_file(m_path), content);
    };

    auto garbage = valid;
    garbage.insert(garbage.end(), 10, 0x5A);
    expect_refused(garbage);  // a header cut short

    auto zeros = valid;
    zeros.insert(zeros.end(), alignment, 0x00);
    expect_refused(zeros);  // a header that is not one

    expect_refused(Bytes(valid.begin(), valid.begin() + 4000));  // cut inside the container

    expect_refused(header_only_container(hsm::BlobMagic::single));  // the first one must be a shared context

    auto misaligned = header_only_container(hsm::BlobMagic::multi);
    const auto second = header_only_container(hsm::BlobMagic::single);
    misaligned.insert(misaligned.end(), second.begin(), second.end());
    expect_refused(misaligned);  // the second one starts at 32
}

TEST_F(HsmMultiBlobWriterTest, a_source_listed_twice_in_the_file_is_refused) {
    Sources sources;
    sources.add(3, 100, 0x03);
    {
        hsm::MultiBlobWriter writer{m_path};
        writer.write_context(sources.context);
    }
    auto bytes = read_file(m_path);
    const Bytes delta(bytes.begin() + alignment, bytes.end());
    bytes.insert(bytes.end(), delta.begin(), delta.end());
    write_file(m_path, bytes);

    EXPECT_THROW(hsm::MultiBlobWriter{m_path}, ov::Exception);
}

// --- Several writers ---

TEST_F(HsmMultiBlobWriterTest, writers_on_one_path_share_what_they_wrote) {
    Sources first;
    first.add(3, 100, 0x03);
    Sources both;
    both.add(3, 100, 0x03);
    both.add(7, 100, 0x07);

    hsm::MultiBlobWriter a{m_path};
    hsm::MultiBlobWriter b{m_path};
    a.write_context(first.context);
    b.write_context(both.context);  // source 3 is already there

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    ASSERT_EQ(containers.size(), 3u);
    EXPECT_EQ(entries_with(containers[2], hsm::weights).size(), 1u);

    a.write_model(11, model_blob());
    EXPECT_THROW(b.write_model(11, model_blob()), ov::Exception);
}

TEST_F(HsmMultiBlobWriterTest, models_written_from_several_threads_all_end_up_indexed) {
    constexpr uint64_t threads = 4;
    constexpr uint64_t per_thread = 6;
    {
        hsm::MultiBlobWriter writer{m_path};
        std::vector<std::thread> pool;
        for (uint64_t t = 0; t < threads; ++t) {
            pool.emplace_back([&writer, t] {
                for (uint64_t i = 0; i < per_thread; ++i) {
                    writer.write_model(t * 100 + i, model_blob());
                }
            });
        }
        for (auto& worker : pool) {
            worker.join();
        }
        writer.close();
    }

    const auto bytes = read_file(m_path);
    const auto containers = parse_file(bytes);
    size_t blobs = 0;
    for (const auto& container : containers) {
        EXPECT_EQ(container.offset % alignment, 0u);
        blobs += container.header.magic == hsm::BlobMagic::single ? 1 : 0;
    }
    EXPECT_EQ(blobs, threads * per_thread);

    hsm::MultiBlobWriter reopened{m_path};
    for (uint64_t t = 0; t < threads; ++t) {
        for (uint64_t i = 0; i < per_thread; ++i) {
            EXPECT_THROW(reopened.write_model(t * 100 + i, model_blob()), ov::Exception);
        }
    }
}

}  // namespace ov::test
