// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#    include <string>
#    include <vector>

#    include "cache_manager.hpp"

namespace ov::storage {

struct Bundle {
    using byte_type = char;
    using tag_type = uint64_t;
    using offset_type = uint64_t;  // should it be ptrdiff or size_t ?
    static_assert(sizeof(byte_type) == 1);

    tag_type tag;
    offset_type length;  // value length in bytes
    // offset to the value from the begining of the stream
    // consider whether it should be relative to this struct position?
    offset_type value_offset;
    // offset_type value_length() const;

    struct BufferView {
        const byte_type* data;
        size_t size;
    };
    std::variant<BufferView, std::istream*, std::vector<byte_type>> value;
    // might be not needed
    offset_type entry_offset;  // offset of the entry from the begining of the stream
    offset_type entry_size;    // total size of the entry in bytes (including tag, length, value_offset, value)
};

class BundlePool {
public:
    using byte_type = Bundle::byte_type;
    using tag_type = Bundle::tag_type;
    using offset_type = Bundle::offset_type;

    // void add_header_entry();
    void add_entry(tag_type tag, std::istream* value, uint64_t value_alignment = 0);
    void add_entry(tag_type tag, uint64_t length, const byte_type* value, uint64_t value_alignment = 0);
    void add_entry(tag_type tag, const std::vector<byte_type>& value, uint64_t value_alignment = 0);
    void add_entry(tag_type tag, std::vector<byte_type>&& value, uint64_t value_alignment = 0);

    void write_to(std::ostream& dest);
    void read_from(std::istream& src);

    // the name reflects that the bundles' values are read as streams
    void read_as_streams_from(std::istream& src);

    const std::vector<Bundle>& pool() const;

private:
    // AccessMode m_access_mode {AccessMode::READ};

    size_t ind_pos{0};

    void append(Bundle&& pack, uint64_t value_alignment);
    std::vector<Bundle> m_pool;

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
