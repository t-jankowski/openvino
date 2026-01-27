// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_manager.hpp"

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
 * @brief This class limits the locale env to a special value in sub-scope
 *
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string newLocale) : m_category(category) {
        m_oldLocale = setlocale(category, nullptr);
        setlocale(m_category, newLocale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_oldLocale.c_str());
    }

private:
    int m_category;
    std::string m_oldLocale;
};

std::filesystem::path FileStorageCacheManager::get_blob_file(const std::string& blob_hash) const {
    return m_cache_path / (blob_hash + ".blob");
}

FileStorageCacheManager::FileStorageCacheManager(std::filesystem::path cache_path)
    : m_cache_path(std::move(cache_path)) {}

FileStorageCacheManager::~FileStorageCacheManager() = default;

void FileStorageCacheManager::write_cache_entry(const std::string& id, StreamWriter writer) {
    // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
    ScopedLocale plocal_C(LC_ALL, "C");
    const auto blob_path = get_blob_file(id);
    std::ofstream stream(blob_path, std::ios_base::binary);
    writer(stream);
    stream.close();
    std::filesystem::permissions(blob_path, std::filesystem::perms::owner_read | std::filesystem::perms::group_read);
}

void FileStorageCacheManager::read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) {
    // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
    ScopedLocale plocal_C(LC_ALL, "C");
    const auto blob_path = get_blob_file(id);
    if (std::filesystem::exists(blob_path)) {
        if (enable_mmap) {
            CompiledBlobVariant compiled_blob{std::in_place_index<0>, ov::read_tensor_data(blob_path)};
            reader(compiled_blob);
        } else {
            std::ifstream stream(blob_path, std::ios_base::binary);
            CompiledBlobVariant compiled_blob{std::in_place_index<1>, std::ref(stream)};
            reader(compiled_blob);
        }
    }
}

void FileStorageCacheManager::remove_cache_entry(const std::string& id) {
    const auto blob_path = get_blob_file(id);

    if (std::filesystem::exists(blob_path)) {
        std::ignore = std::filesystem::remove(blob_path);
    }
}
}  // namespace ov
