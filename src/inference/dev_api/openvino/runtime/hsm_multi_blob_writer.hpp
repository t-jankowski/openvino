// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icache_manager.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief Writes a multi-blob file: HSM containers back to back, each starting at a #multi_blob_alignment boundary.
 * The first container is always a shared-context one (#BlobMagic::multi), possibly empty; model blobs
 * (#BlobMagic::single, written by plugins) and further shared-context containers follow. Nothing already in the
 * file is ever modified.
 *
 * @note Opening a path that is missing or empty creates the file; otherwise the file is scanned and must be valid
 * (a broken tail is an error, never repaired). Writers on the same path within a process share one state and
 * are safe to use from several threads; several processes on one file are not supported.
 * @note A failed append is cut back, leaving the file as it was.
 * @note Releasing the last writer of a path writes the model records still held back, ignoring any failure - call
 * #close() to learn about one.
 */
class OPENVINO_RUNTIME_API MultiBlobWriter {
public:
    /// @throws ov::Exception if the file cannot be created, or exists but is not a valid multi-blob file.
    explicit MultiBlobWriter(std::filesystem::path path);

    ~MultiBlobWriter();

    MultiBlobWriter(const MultiBlobWriter&) = delete;
    MultiBlobWriter& operator=(const MultiBlobWriter&) = delete;
    MultiBlobWriter(MultiBlobWriter&&) = delete;
    MultiBlobWriter& operator=(MultiBlobWriter&&) = delete;

    /**
     * @brief Appends what @p context holds and the file doesn't know yet: the buffers of new sources, the constants
     * of known sources, and the model records held back by #write_model(). Does nothing if there is none.
     * @note Constants of a source that is neither in the file nor in @p context are left out; expired buffers
     * and runtime-only sources are not written.
     */
    void write_context(const ov::wsh::Context& context);

    /**
     * @brief Appends the model blob @p writer produces, which must be exactly one #BlobMagic::single container
     * written at the stream position it is given. The blob is indexed by the next #write_context() or #close().
     * @param model_id Id of the model (the cache blob id); must not be in the file yet.
     */
    void write_model(uint64_t model_id, const ov::ICacheManager::StreamWriter& writer);

    /// Placeholder for sources living outside the file - accepted and ignored.
    void register_external_source(ov::wsh::DataID source_id, ov::util::MemoryView descriptor);

    /// Writes the model records still held back. Unlike the destructor, reports a failure.
    void close();

private:
    /// Per-file state shared by every writer of the same path within the process (pimpl).
    struct FileState;
    std::shared_ptr<FileState> m_file_state;
};

}  // namespace v1
}  // namespace ov::runtime::hsm
