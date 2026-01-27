// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

/**
 * @brief This class represents private interface for Cache Manager
 *
 */
class ICacheManager {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when OpenVINO intends to write model to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, model will not be cached
     *
     * @param id Id of cache (hash of the model)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void write_cache_entry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Variant type for compiled blob representation
     */
    using CompiledBlobVariant = std::variant<const ov::Tensor, std::reference_wrapper<std::istream>>;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(CompiledBlobVariant&)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param enable_mmap use mmap or ifstream to read model file
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) = 0;

    /**
     * @brief Callback when OpenVINO intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the model)
     */
    virtual void remove_cache_entry(const std::string& id) = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models.
 *
 */
class FileStorageCacheManager final : public ICacheManager {
public:
    FileStorageCacheManager(std::filesystem::path cache_path);
    ~FileStorageCacheManager() override;

private:
    std::filesystem::path m_cache_path;

    std::filesystem::path get_blob_file(const std::string& blob_hash) const;

    void write_cache_entry(const std::string& id, StreamWriter writer) override;
    void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) override;
    void remove_cache_entry(const std::string& id) override;
};

}  // namespace ov
