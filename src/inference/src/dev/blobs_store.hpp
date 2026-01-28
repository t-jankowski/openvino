// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef BLOBS_STORE_ENABLED
#    include <string>
#    include <vector>

#    include "cache_manager.hpp"

namespace ov::storage {

// rename TLVPack to sth meaningful
struct TLVPack {
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
    const byte_t* value_outside{nullptr};
    // either raw external or inside vector or stream, not both .. pointer takes priority
    std::vector<byte_t> value_inside;
    std::stringstream value_stream;

    // might be not needed
    uint64_t entry_offset;  // offset of the entry from the begining of the stream
    uint64_t entry_size;    // total size of the entry in bytes (including tag, length, value_offset, value)
};

class Blobs {
public:
    // void add_header_entry();
    void add_entry(TLVPack::tag_t tag, std::stringstream&& value, uint64_t value_alignment = 0);
    void add_entry(TLVPack::tag_t tag, uint64_t length, const TLVPack::byte_t* value, uint64_t value_alignment = 0);
    void add_entry(TLVPack::tag_t tag, const std::vector<TLVPack::byte_t>& value, uint64_t value_alignment = 0);
    void add_entry(TLVPack::tag_t tag, std::vector<TLVPack::byte_t>&& value, uint64_t value_alignment = 0);
    void write_to(std::ostream& dest);

private:
    // AccessMode m_access_mode {AccessMode::READ};

    size_t ind_pos{0};

    void append(TLVPack&& pack, uint64_t value_alignment);
    std::vector<TLVPack> m_entries;

    // alignment of value e.g. per page size 4096
    const uint64_t m_default_value_alignment{1};
};

class BlobsCacheEmulation : public ICacheManager {
public:
    // Single file for now .. but consider its extendability to multiple files `std::map<std::filesystem::path, Blobs>
    // m_storage;`
    BlobsCacheEmulation(std::filesystem::path blobs_path);
    void write_cache_entry(const std::string& id, StreamWriter writer) override;

    void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) override;

    void remove_cache_entry(const std::string& id) override;

private:
    std::filesystem::path m_blobs_path;
    Blobs m_blobs;

    std::map<std::string, std::stringstream> m_stream_writers;
};
}  // namespace ov::storage
#endif  // BLOBS_STORE_ENABLED
