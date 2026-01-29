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

    tag_t tag;
    uint64_t length;  // value length in bytes
    // offset to the value from the begining of the stream
    // todo: consider whether it should be relative to this struct position?
    uint64_t value_offset;  // should it be ptrdiff or size_t ?

    static constexpr size_t fixed_size() {
        return sizeof(tag) + sizeof(length) + sizeof(value_offset);
    }

    struct BufferView {
        byte_t* data;
        size_t size;
    };
    std::variant<BufferView, std::stringstream*, std::vector<byte_t>> value;
    // might be not needed
    uint64_t entry_offset;  // offset of the entry from the begining of the stream
    uint64_t entry_size;    // total size of the entry in bytes (including tag, length, value_offset, value)
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
