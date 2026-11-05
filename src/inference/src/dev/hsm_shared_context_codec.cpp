// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_shared_context_codec.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

namespace ov::runtime::hsm {
inline namespace v1 {

namespace {

constexpr size_t record_header_size = sizeof(uint8_t) + sizeof(uint32_t);
constexpr size_t source_body_size = 2 * sizeof(uint64_t);
constexpr size_t constant_body_size = 4 * sizeof(uint64_t) + sizeof(uint8_t);
constexpr size_t model_body_size = 2 * sizeof(uint64_t);

template <class T>
void put(std::vector<std::byte>& out, T value) {
    const auto at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &value, sizeof(T));
}

template <class Map>
std::vector<typename Map::key_type> sorted_keys(const Map& map) {
    std::vector<typename Map::key_type> keys;
    keys.reserve(map.size());
    for (const auto& entry : map) {
        keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Builds one record in `record` and hands it to `sink`; `fill` appends exactly `body_size` bytes.
template <class Fill>
void emit(const SectionSink& sink,
          std::vector<std::byte>& record,
          SharedContextRecord kind,
          size_t body_size,
          Fill&& fill) {
    record.clear();
    put(record, static_cast<uint8_t>(kind));
    put(record, static_cast<uint32_t>(body_size));
    fill(record);
    sink({record.data(), record.size()});
}

// Bounds-checked reader: every access states how many bytes it needs, so a corrupt length never reads past the end.
class Cursor {
public:
    explicit Cursor(ov::util::MemoryView view) noexcept : m_at{view.data()}, m_left{view.size()} {}

    bool empty() const noexcept {
        return m_left == 0;
    }

    template <class T>
    bool read(T& value) noexcept {
        if (m_left < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, m_at, sizeof(T));
        m_at += sizeof(T);
        m_left -= sizeof(T);
        return true;
    }

    std::optional<ov::util::MemoryView> take(size_t size) noexcept {
        if (m_left < size) {
            return std::nullopt;
        }
        const ov::util::MemoryView taken{m_at, size};
        m_at += size;
        m_left -= size;
        return taken;
    }

private:
    const std::byte* m_at;
    size_t m_left;
};

// A 32-bit size_t can't represent every u64 on the wire.
constexpr bool fits_size_t(uint64_t value) noexcept {
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        return value <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    } else {
        return true;
    }
}

bool decode_source(ov::util::MemoryView body, SharedContextPayload& payload) noexcept {
    uint64_t source_id = 0;
    OffsetType offset = 0;
    Cursor in{body};
    return body.size() == source_body_size && in.read(source_id) && in.read(offset) &&
           payload.sources.emplace(source_id, offset).second;
}

bool decode_constant(ov::util::MemoryView body, SharedContextPayload& payload) noexcept {
    uint64_t source_id = 0, constant_id = 0, offset = 0, size = 0;
    uint8_t type = 0;
    Cursor in{body};
    if (body.size() != constant_body_size || !in.read(source_id) || !in.read(constant_id) || !in.read(offset) ||
        !in.read(size) || !in.read(type)) {
        return false;
    }
    if (type > static_cast<uint8_t>(ov::element::Type_t::f8e8m0) || !fits_size_t(offset) || !fits_size_t(size)) {
        return false;
    }
    const ov::wsh::WeightMetaData meta{static_cast<size_t>(offset),
                                       static_cast<size_t>(size),
                                       ov::element::Type{static_cast<ov::element::Type_t>(type)}};
    return payload.constants[source_id].emplace(constant_id, meta).second;
}

bool decode_model(ov::util::MemoryView body, SharedContextPayload& payload) noexcept {
    uint64_t blob_id = 0;
    OffsetType offset = 0;
    Cursor in{body};
    return body.size() == model_body_size && in.read(blob_id) && in.read(offset) &&
           payload.models.emplace(blob_id, offset).second;
}

// An unknown or reserved kind is skipped, not rejected: new kinds must stay additive.
bool decode_record(SharedContextRecord kind, ov::util::MemoryView body, SharedContextPayload& payload) noexcept {
    switch (kind) {
    case SharedContextRecord::source:
        return decode_source(body, payload);
    case SharedContextRecord::constant:
        return decode_constant(body, payload);
    case SharedContextRecord::model:
        return decode_model(body, payload);
    default:
        return true;
    }
}

}  // namespace

size_t SharedContextEncoder::size() const noexcept {
    size_t constants = 0;
    for (const auto& entry : m_payload.constants) {
        constants += entry.second.size();
    }
    return m_payload.sources.size() * (record_header_size + source_body_size) +
           constants * (record_header_size + constant_body_size) +
           m_payload.models.size() * (record_header_size + model_body_size);
}

void SharedContextEncoder::encode(const SectionSink& sink) const {
    std::vector<std::byte> record;
    record.reserve(record_header_size + constant_body_size);

    for (const auto source_id : sorted_keys(m_payload.sources)) {
        emit(sink, record, SharedContextRecord::source, source_body_size, [&](auto& out) {
            put<uint64_t>(out, source_id);
            put<uint64_t>(out, m_payload.sources.at(source_id));
        });
    }
    for (const auto source_id : sorted_keys(m_payload.constants)) {
        const auto& metas = m_payload.constants.at(source_id);
        for (const auto constant_id : sorted_keys(metas)) {
            const auto& meta = metas.at(constant_id);
            emit(sink, record, SharedContextRecord::constant, constant_body_size, [&](auto& out) {
                put<uint64_t>(out, source_id);
                put<uint64_t>(out, constant_id);
                put<uint64_t>(out, meta.m_offset);
                put<uint64_t>(out, meta.m_size);
                put<uint8_t>(out, static_cast<uint8_t>(static_cast<ov::element::Type_t>(meta.m_type)));
            });
        }
    }
    for (const auto blob_id : sorted_keys(m_payload.models)) {
        emit(sink, record, SharedContextRecord::model, model_body_size, [&](auto& out) {
            put<uint64_t>(out, blob_id);
            put<uint64_t>(out, m_payload.models.at(blob_id));
        });
    }
}

std::optional<SharedContextPayload> decode_shared_context(ov::util::MemoryView bytes) noexcept {
    try {
        SharedContextPayload payload;
        Cursor in{bytes};
        while (!in.empty()) {
            uint8_t kind = 0;
            uint32_t body_size = 0;
            if (!in.read(kind) || !in.read(body_size)) {
                return std::nullopt;
            }
            const auto body = in.take(body_size);
            if (!body || !decode_record(static_cast<SharedContextRecord>(kind), *body, payload)) {
                return std::nullopt;
            }
        }
        return payload;
    } catch (const std::bad_alloc&) {  // the only way building the maps can fail
        return std::nullopt;
    }
}

std::optional<SharedContextPayload> decode_shared_context(const Section& section) noexcept {
    try {
        if (section.size() == 0) {
            return SharedContextPayload{};
        } else if (const auto view = section.view(); !view.empty()) {
            return decode_shared_context(view);
        } else if (const auto bytes = section.to_bytes()) {
            return decode_shared_context(ov::util::MemoryView{bytes->data(), bytes->size()});
        } else {
            return std::nullopt;
        }
    } catch (const std::exception&) {  // bad_alloc or a stream configured to throw
        return std::nullopt;
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
