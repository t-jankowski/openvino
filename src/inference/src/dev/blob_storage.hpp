// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef BLOBS_STORAGE_ENABLED
#    include <string>
#    include <vector>

#    include "cache_manager.hpp"

namespace ov::storage {

struct Bundle {
    using byte_t = char;
    static_assert(sizeof(byte_t) == 1);
    using tag_t = uint64_t;
    using offset_t = uint64_t;

    tag_t tag;
    offset_t length;  // value length in bytes
    // offset to the value from the begining of the stream
    // consider whether it should be relative to this struct position?
    offset_t value_offset;  // should it be ptrdiff or size_t ?
    offset_t value_length() const;

    struct BufferView {
        const byte_t* data;
        size_t size;
    };
    std::variant<BufferView, std::stringstream*, std::vector<byte_t>> value;
    // might be not needed
    offset_t entry_offset;  // offset of the entry from the begining of the stream
    offset_t entry_size;    // total size of the entry in bytes (including tag, length, value_offset, value)
};

class BundlePool {
public:
    using tag_t = Bundle::tag_t;
    using byte_t = Bundle::byte_t;
    // void add_header_entry();
    void add_entry(tag_t tag, std::stringstream* value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, uint64_t length, const byte_t* value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, const std::vector<byte_t>& value, uint64_t value_alignment = 0);
    void add_entry(tag_t tag, std::vector<byte_t>&& value, uint64_t value_alignment = 0);

    void write_to(std::ostream& dest);
    void read_from(std::istream& src);

    const std::vector<Bundle>& entries() const;

private:
    // AccessMode m_access_mode {AccessMode::READ};

    size_t ind_pos{0};

    void append(Bundle&& pack, uint64_t value_alignment);
    std::vector<Bundle> m_entries;

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

    std::map<std::string, std::stringstream> m_blob_streams;

    void write_to_file();
    void read_from_file();
};
}  // namespace ov::storage
#endif  // BLOBS_STORAGE_ENABLED
