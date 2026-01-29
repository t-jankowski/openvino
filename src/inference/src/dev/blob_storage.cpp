// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef BLOBS_STORAGE_ENABLED
#    include "blob_storage.hpp"

#    include <cassert>
// #    include <utility>

namespace ov::storage {

struct Pack {
    using byte_t = char;
    static_assert(sizeof(byte_t) == 1);
    using tag_t = uint64_t;

    tag_t tag;
    uint64_t length;  // value length in bytes
    // offset to the value from the begining of the stream
    uint64_t value_offset;  // should it be ptrdiff or size_t ?

    static constexpr size_t fixed_size() {
        return sizeof(tag) + sizeof(length) + sizeof(value_offset);
    }

    /// Pointer to an external buffer containing raw 8-bit values managed outside the current scope.
    byte_t* value_outside{nullptr};
    // either raw external buffer or external stream or inside vector , not both .. pointer takes priority
    std::stringstream* value_stream{nullptr};
    std::vector<byte_t> value_inside;

    // might be not needed
    uint64_t entry_offset;  // offset of the entry from the begining of the stream
    uint64_t entry_size;    // total size of the entry in bytes (including tag, length, value_offset, value)
};

class Bundle {
public:
    using tag_t = Pack::tag_t;
    using byte_t = Pack::byte_t;
    // void add_header_entry();
    void add_entry(tag_t tag, std::stringstream* value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, uint64_t length, byte_t* value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, const std::vector<byte_t>& value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, std::vector<byte_t>&& value, uint64_t value_alignment = 0);
    void write_to(std::ostream& dest);

private:
    // AccessMode m_access_mode {AccessMode::READ};

    size_t ind_pos{0};

    void append(Pack&& pack, uint64_t value_alignment);
    std::vector<Pack> m_entries;

    // alignment of value e.g. per page size 4096
    const uint64_t m_default_value_alignment{1};
};

void Bundle::add_entry(tag_t tag, std::stringstream* value, uint64_t value_alignment) {
    Pack pack;
    pack.tag = tag;
    value->seekg(0, std::ios::end);
    pack.length = value->tellg();
    value->seekg(0, std::ios::beg);
    pack.value_stream = value;
    append(std::move(pack), value_alignment);
}

void Bundle::add_entry(tag_t tag, uint64_t length, byte_t* value, uint64_t value_alignment) {
    Pack pack;
    pack.tag = tag;
    pack.length = length;
    pack.value_outside = value;
    append(std::move(pack), value_alignment);
}

void Bundle::add_entry(tag_t tag, const std::vector<byte_t>& value, uint64_t value_alignment) {
    Pack pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value_inside = value;
    append(std::move(pack), value_alignment);
}

void Bundle::add_entry(tag_t tag, std::vector<byte_t>&& value, uint64_t value_alignment) {
    Pack pack;
    pack.tag = tag;
    pack.length = value.size();
    pack.value_inside = std::move(value);
    append(std::move(pack), value_alignment);
}

void Bundle::write_to(std::ostream& dest) {
    for (const auto& pack : m_entries) {
        dest.write(reinterpret_cast<const char*>(&pack.tag), sizeof(pack.tag));
        dest.write(reinterpret_cast<const char*>(&pack.length), sizeof(pack.length));
        dest.write(reinterpret_cast<const char*>(&pack.value_offset), sizeof(pack.value_offset));
        // write padding if any
        auto current_pos = dest.tellp();
        if (current_pos < static_cast<std::streampos>(pack.value_offset)) {
            const size_t padding_size = pack.value_offset - current_pos;
            const std::vector<byte_t> padding(padding_size, 0);
            dest.write(padding.data(), static_cast<std::streamsize>(padding_size));
        }
        // write value
        if (pack.value_outside) {
            dest.write(pack.value_outside, static_cast<std::streamsize>(pack.length));
        } else if (pack.value_stream) {
            dest << pack.value_stream->rdbuf();
        } else {
            dest.write(pack.value_inside.data(), static_cast<std::streamsize>(pack.value_inside.size()));
        }
    }
}

void Bundle::append(Pack&& pack, uint64_t value_alignment) {
    if (value_alignment == 0) {
        value_alignment = m_default_value_alignment;
    }

    pack.entry_offset = ind_pos;
    pack.value_offset = pack.entry_offset + Pack::fixed_size() + value_alignment - 1;
    pack.value_offset -= pack.value_offset % value_alignment;
    pack.entry_size = (pack.value_offset - pack.entry_offset) + pack.length;
    ind_pos += pack.entry_size;
    m_entries.push_back(std::move(pack));
}

BlobsCacheEmulation::BlobsCacheEmulation(std::filesystem::path blobs_path) : m_blobs_path(std::move(blobs_path)) {}

enum class Tag : Bundle::tag_t {
    Header = 0x0100,
    SharedContext = 0x0101,
    ContentSummary = 0x01000,
    Id = 0x01001,
    Blob = 0x01002,
};

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
    Bundle blobs;

    // blobs add header
    // blobs add shared context
    // blobs add content summary

    for (auto& [id, stream] : m_blob_streams) {
        blobs.add_entry(static_cast<Bundle::tag_t>(Tag::Id), std::vector<Bundle::byte_t>(id.begin(), id.end()));
        blobs.add_entry(static_cast<Bundle::tag_t>(Tag::Blob), &stream);
    }
    blobs.write_to(blobs_file);
    blobs_file.close();
}

void BlobsCacheEmulation::read_from_file() {
    // implementation here
}
}  // namespace ov::storage
#endif  // BLOBS_STORAGE_ENABLED
