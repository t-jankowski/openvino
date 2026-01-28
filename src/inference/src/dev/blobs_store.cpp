// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef BLOBS_STORE_ENABLED
#    include "blobs_store.hpp"

#    include <cassert>
// #    include <utility>

namespace ov::storage {

// void Blobs::add_header_entry() {
//     assert(ind_pos == 0);
//     std::string header{"OpenVINO Blobs Store POC v0.1"};
//     TLVPack pack;
//     pack.tag = 0x100;
//     pack.length = header.size();
//     pack.value_inside = std::vector<TLVPack::byte_t>(header.begin(), header.end());
//     append(std::move(pack), 1);
// }

void Blobs::add_entry(TLVPack::tag_t tag, std::stringstream&& value, uint64_t value_alignment) {
    TLVPack pack;
    pack.tag = tag;
    value.seekg(0, std::ios::end);
    pack.length = value.tellg();
    value.seekg(0, std::ios::beg);
    pack.value_stream = std::move(value);
    append(std::move(pack), value_alignment);
}

void Blobs::add_entry(TLVPack::tag_t tag, uint64_t length, const TLVPack::byte_t* value, uint64_t value_alignment) {
    TLVPack pack;
    pack.tag = tag;
    pack.length = length;
    pack.value_outside = value;
    append(std::move(pack), value_alignment);
}

void Blobs::add_entry(TLVPack::tag_t tag, const std::vector<TLVPack::byte_t>& value, uint64_t value_alignment) {
    TLVPack pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value_inside = value;
    append(std::move(pack), value_alignment);
}

void Blobs::add_entry(TLVPack::tag_t tag, std::vector<TLVPack::byte_t>&& value, uint64_t value_alignment) {
    TLVPack pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value_inside = std::move(value);
    append(std::move(pack), value_alignment);
}

void Blobs::write_to(std::ostream& dest) {
    for (const auto& pack : m_entries) {
        dest.write(reinterpret_cast<const char*>(&pack.tag), sizeof(pack.tag));
        dest.write(reinterpret_cast<const char*>(&pack.length), sizeof(pack.length));
        dest.write(reinterpret_cast<const char*>(&pack.value_offset), sizeof(pack.value_offset));
        // write padding if any
        auto current_pos = dest.tellp();
        if (current_pos < static_cast<std::streampos>(pack.value_offset)) {
            size_t padding_size = pack.value_offset - current_pos;
            std::vector<TLVPack::byte_t> padding(padding_size, 0);
            dest.write(padding.data(), padding.size());
        }
        // write value
        if (pack.value_outside) {
            dest.write(pack.value_outside, pack.length);
        } else {
            dest.write(pack.value_inside.data(), pack.value_inside.size());
        }
    }
}

void Blobs::append(TLVPack&& pack, uint64_t value_alignment) {
    if (value_alignment == 0) {
        value_alignment = m_default_value_alignment;
    }

    pack.entry_offset = ind_pos;
    pack.value_offset = pack.entry_offset + TLVPack::fixed_size() + value_alignment - 1;
    pack.value_offset -= pack.value_offset % value_alignment;
    pack.entry_size = (pack.value_offset - pack.entry_offset) + pack.length;
    ind_pos += pack.entry_size;
    m_entries.push_back(std::move(pack));
}

BlobsCacheEmulation::BlobsCacheEmulation(std::filesystem::path blobs_path) : m_blobs_path(std::move(blobs_path)) {}

enum class Tag : TLVPack::tag_t {
    Id = 0x01001,
    Blob = 0x01002,
};
void BlobsCacheEmulation::write_cache_entry(const std::string& id, StreamWriter writer) {
    m_stream_writers[id] = std::stringstream{};
    writer(m_stream_writers[id]);

    m_blobs.add_entry(static_cast<TLVPack::tag_t>(Tag::Id), std::vector<TLVPack::byte_t>(id.begin(), id.end()));
    m_blobs.add_entry(static_cast<TLVPack::tag_t>(Tag::Blob), std::move(m_stream_writers[id]));
}
void BlobsCacheEmulation::read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) {
    // implementation here
}
void BlobsCacheEmulation::remove_cache_entry(const std::string& id) {
    // implementation here
}

}  // namespace ov::storage
#endif  // BLOBS_STORE_ENABLED
