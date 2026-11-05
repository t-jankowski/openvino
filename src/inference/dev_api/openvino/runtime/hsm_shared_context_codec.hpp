// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Codec of the #shared_context_tag section - the index kept in every #BlobMagic::multi container of a
 * multi-blob file.
 * @file openvino/runtime/hsm_shared_context_codec.hpp
 *
 * Payload layout (native byte order, like the rest of the format): a flat sequence of records, no leading count.
 * @verbatim
   record          kind u8 | body_size u32 | body
   source          body: source_id u64 | offset u64
   constant        body: source_id u64 | constant_id u64 | offset u64 | size u64 | element type u8
   model           body: model_id u64 | offset u64
   external_source reserved - never written, skipped when read
   @endverbatim
 * Source and model offsets are measured from the first byte of the multi-blob file; constant offsets from the first
 * byte of their source. Records are written ordered by kind, then by key.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/// Weight source id -> file offset of the #weights_tag section holding its bytes (any container of the file).
using SourceLinks = std::unordered_map<ov::wsh::DataID, OffsetType>;

/// Model id (the cache blob id) -> file offset of its #BlobMagic::single container.
using ModelIndex = std::unordered_map<uint64_t, OffsetType>;

/// Placeholder for sources living outside the file (external buffers).
using ExternalSources = std::unordered_map<ov::wsh::DataID, std::vector<std::byte>>;

/// Everything a shared-context container tells about the file, independent of how it is encoded.
struct SharedContextPayload {
    SourceLinks sources;                //!< Sources introduced by the container.
    ov::wsh::WeightRegistry constants;  //!< Constants of the sources, keyed by source id and constant id.
    ModelIndex models;                  //!< Model blobs the container makes selectable.
    ExternalSources external_sources;   //!< Reserved: neither encoded nor decoded yet.
};

/// Record kinds on the wire - explicit values, never reuse or change one once shipped.
enum class SharedContextRecord : uint8_t {
    source = 1,
    constant = 2,
    model = 3,
    external_source = 4,  //!< Reserved for sources outside the file; skipped by the decoder.
};

/**
 * @brief Writes a #SharedContextPayload as the content of a #shared_context_tag section.
 * @note Borrows the payload, which must outlive #IWriter::finalize() - a temporary is rejected at compile time.
 */
class OPENVINO_RUNTIME_API SharedContextEncoder final : public ISectionEncoder {
public:
    explicit SharedContextEncoder(const SharedContextPayload& payload) noexcept : m_payload{payload} {}
    explicit SharedContextEncoder(const SharedContextPayload&&) = delete;

    /// Exact number of bytes #encode() produces - lets the sized #IWriter::add_section() overload be used.
    size_t size() const noexcept;

    void encode(const SectionSink& sink) const override;

private:
    const SharedContextPayload& m_payload;
};

/**
 * @brief Reads a #SharedContextPayload from the content of a #shared_context_tag section.
 * @return `std::nullopt` if the bytes are malformed: truncated, a known record of the wrong size, an unknown element
 * type, a value that doesn't fit `size_t`, or the same key twice. Records of an unknown kind are skipped.
 */
OPENVINO_RUNTIME_API std::optional<SharedContextPayload> decode_shared_context(ov::util::MemoryView bytes) noexcept;

/// @overload Reads the section through #Section::view() when it can, otherwise copies it once.
OPENVINO_RUNTIME_API std::optional<SharedContextPayload> decode_shared_context(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
