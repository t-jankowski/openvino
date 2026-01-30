// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "blob_storage.hpp"

#include <cassert>

#include "openvino/util/variant_visitor.hpp"

namespace ov::storage {

// Bundle::offset_type Bundle::value_length() const {
//     return length;
// }

enum class Tag : Bundle::tag_type {
    header = 0x0100,
    shared_context = 0x0101,
    content_summary = 0x0102,
    blob = 0x1000,
    blob_id = 0x1001,
    blob_data = 0x1002,
};

void BundlePool::add_entry(tag_type tag, std::istream* value, uint64_t value_alignment) {
    Bundle pack;
    pack.tag = tag;
    value->seekg(0, std::ios::end);
    pack.length = value->tellg();
    value->seekg(0, std::ios::beg);
    pack.value = value;
    append(std::move(pack), value_alignment);
}

void BundlePool::add_entry(tag_type tag, uint64_t length, const byte_type* value, uint64_t value_alignment) {
    Bundle pack;
    pack.tag = tag;
    pack.length = length;
    pack.value = Bundle::BufferView{value, length};
    append(std::move(pack), value_alignment);
}

void BundlePool::add_entry(tag_type tag, const std::vector<byte_type>& value, uint64_t value_alignment) {
    Bundle pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value = value;
    append(std::move(pack), value_alignment);
}

void BundlePool::add_entry(tag_type tag, std::vector<byte_type>&& value, uint64_t value_alignment) {
    Bundle pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value = std::move(value);
    append(std::move(pack), value_alignment);
}

const std::vector<Bundle>& BundlePool::pool() const {
    return m_pool;
}

void BundlePool::write_to(std::ostream& dest) {
    for (const auto& pack : m_pool) {
        dest.write(reinterpret_cast<const char*>(&pack.tag), sizeof(pack.tag));
        dest.write(reinterpret_cast<const char*>(&pack.length), sizeof(pack.length));
        dest.write(reinterpret_cast<const char*>(&pack.value_offset), sizeof(pack.value_offset));
        // write padding if any
        auto current_pos = dest.tellp();
        if (current_pos < static_cast<std::streampos>(pack.value_offset)) {
            const size_t padding_size = pack.value_offset - current_pos;
            const std::vector<byte_type> padding(padding_size, 0);
            dest.write(padding.data(), static_cast<std::streamsize>(padding_size));
        }
        // write value
        ov::util::VariantVisitor value_to_stream{[&dest](const Bundle::BufferView& buffer_view) {
                                                     dest.write(buffer_view.data,
                                                                static_cast<std::streamsize>(buffer_view.size));
                                                 },
                                                 [&dest](std::istream* stream) {
                                                     dest << stream->rdbuf();
                                                 },
                                                 [&dest](const std::vector<byte_type>& data) {
                                                     dest.write(data.data(), static_cast<std::streamsize>(data.size()));
                                                 }};
        std::visit(value_to_stream, pack.value);
    }
}

void BundlePool::read_from(std::istream& src) {
    m_pool.clear();
    ind_pos = 0;
    while (src.peek() != EOF) {
        Bundle pack;
        src.read(reinterpret_cast<char*>(&pack.tag), sizeof(pack.tag));
        src.read(reinterpret_cast<char*>(&pack.length), sizeof(pack.length));
        src.read(reinterpret_cast<char*>(&pack.value_offset), sizeof(pack.value_offset));
        pack.entry_offset = ind_pos;
        pack.entry_size = (pack.value_offset - pack.entry_offset) + pack.length;
        ind_pos += pack.entry_size;

        // move to value offset
        src.seekg(pack.value_offset, std::ios::beg);
        // read value
        std::vector<byte_type> buffer(pack.length);
        src.read(buffer.data(), static_cast<std::streamsize>(pack.length));
        pack.value = std::move(buffer);
        m_pool.push_back(std::move(pack));
    }
}

void BundlePool::read_as_streams_from(std::istream& src) {}

void BundlePool::append(Bundle&& pack, uint64_t value_alignment) {
    if (value_alignment == 0) {
        value_alignment = m_default_value_alignment;
    }

    // ! move it write_to method
    // Assumed invariant order of bundles to store. If needed otherwise the offset calculation should go to write
    // method.
    pack.entry_offset = ind_pos;
    constexpr uint64_t header_size = sizeof(pack.tag) + sizeof(pack.length) + sizeof(pack.value_offset);
    pack.value_offset = pack.entry_offset + header_size + value_alignment - 1;
    pack.value_offset -= pack.value_offset % value_alignment;
    pack.entry_size = (pack.value_offset - pack.entry_offset) + pack.length;
    ind_pos += pack.entry_size;
    m_pool.push_back(std::move(pack));
}

BlobsCacheEmulation::BlobsCacheEmulation(std::filesystem::path blobs_path) : m_blobs_path(std::move(blobs_path)) {}

void BlobsCacheEmulation::write_cache_entry(const std::string& id, StreamWriter writer) {
    m_blob_streams[id] = std::stringstream{};
    writer(m_blob_streams[id]);
    write_to_file();
}

void BlobsCacheEmulation::read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) {
    // implementation here
}

void BlobsCacheEmulation::remove_cache_entry(const std::string& id) {
    m_blob_streams.erase(id);
    write_to_file();
}

void BlobsCacheEmulation::write_to_file() {
    std::ofstream blobs_file(m_blobs_path, std::ios::binary);
    BundlePool blobs;

    // blobs add header
    // blobs add shared context
    // blobs add content summary

    for (auto& [id, stream] : m_blob_streams) {
        blobs.add_entry(static_cast<BundlePool::tag_type>(Tag::blob_id),
                        std::vector<BundlePool::byte_type>(id.begin(), id.end()));
        blobs.add_entry(static_cast<BundlePool::tag_type>(Tag::blob_data), &stream);
    }
    blobs.write_to(blobs_file);
    blobs_file.close();
}

void BlobsCacheEmulation::read_from_file() {
    // implementation here
}
}  // namespace ov::storage
