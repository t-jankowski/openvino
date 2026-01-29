// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef BLOBS_STORAGE_ENABLED
#    include <string>
#    include <vector>

#    include "cache_manager.hpp"

namespace ov::storage {

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
