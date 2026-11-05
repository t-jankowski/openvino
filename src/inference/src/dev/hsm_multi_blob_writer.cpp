// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_multi_blob_writer.hpp"

#include <algorithm>
#include <array>
#include <clocale>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_shared_context_codec.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

namespace {

constexpr uint64_t align_up(uint64_t value) noexcept {
    return (value + multi_blob_alignment - 1) / multi_blob_alignment * multi_blob_alignment;
}

void write_zeros(std::ostream& stream, uint64_t count) {
    static const std::array<char, multi_blob_alignment> zeros{};
    while (count > 0 && stream) {
        const auto chunk = std::min<uint64_t>(count, zeros.size());
        stream.write(zeros.data(), static_cast<std::streamsize>(chunk));
        count -= chunk;
    }
}

// Links are promised before the container exists, so check the writer really placed the sources there.
void verify_weights_offsets(std::fstream& stream, uint64_t start, const std::vector<OffsetType>& promised) {
    const auto resume = stream.tellp();
    stream.flush();

    Header header{};
    stream.seekg(static_cast<std::streamoff>(start));
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    OPENVINO_ASSERT(stream && is_valid_header_fields(header), "Cannot read back the shared-context container");

    std::vector<ManifestEntry> manifest(static_cast<size_t>(header.manifest_size / sizeof(ManifestEntry)));
    if (!manifest.empty()) {
        stream.seekg(static_cast<std::streamoff>(start + header.manifest_offset));
        stream.read(reinterpret_cast<char*>(manifest.data()), static_cast<std::streamsize>(header.manifest_size));
        OPENVINO_ASSERT(stream, "Cannot read back the manifest of the shared-context container");
    }

    size_t next = 0;
    for (const auto& entry : manifest) {
        if (entry.device == any_device_id && entry.tag.id() == weights) {
            OPENVINO_ASSERT(next < promised.size() && start + entry.offset == promised[next],
                            "A weights section was not placed where its link points");
            ++next;
        }
    }
    OPENVINO_ASSERT(next == promised.size(), "A weights section is missing from the shared-context container");
    stream.seekp(resume);
}

}  // namespace

// State of one multi-blob file. All changes to the file are made through it.
// There is one instance per file path in the process. Writers opened on the same path get the same instance
// from a static registry, so they append to the file one after another and never overwrite each other.
// It stores:
//  - the end of the last container in the file,
//  - the sources, constants and models that are already in the file,
//  - the models whose blob is already written, but whose index record is not (pending models).
// The mutex protects all of this data.
// A model blob is written at once. Its index record is written later, in the next shared-context container.
// When the last writer of the path is released, the records still waiting are written too.
struct MultiBlobWriter::FileState {
    explicit FileState(std::filesystem::path file) : m_path{std::move(file)} {}

    std::mutex m_mutex;
    const std::filesystem::path m_path;
    uint64_t m_end = 0;               //!< End of the last container; where the next one is placed (unaligned).
    SourceLinks m_sources;            //!< Every source in the file.
    wsh::WeightRegistry m_constants;  //!< Every constant in the file.
    ModelIndex m_models;              //!< Models with an index record.
    ModelIndex m_pending_models;      //!< Model blobs written, index record not yet.

    static std::mutex& registry_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static auto& registry() {
        static std::map<std::filesystem::path, std::weak_ptr<FileState>> entries;
        return entries;
    }

    static std::shared_ptr<FileState> acquire(const std::filesystem::path& file) {
        // absolute() first: weakly_canonical() leaves a relative path to a file that doesn't exist yet relative, so
        // the same file would get two keys once it exists.
        const auto key = std::filesystem::weakly_canonical(std::filesystem::absolute(file));
        std::lock_guard<std::mutex> lock{registry_mutex()};
        util::erase_if(registry(), [](const auto& entry) {
            return entry.second.expired();
        });
        if (const auto found = registry().find(key); found != registry().end()) {
            if (auto live = found->second.lock()) {
                return live;
            }
        }
        auto fresh = std::make_shared<FileState>(key);
        fresh->initialize();
        registry()[key] = fresh;
        return fresh;
    }

    // The last writer of a path flushes while still holding the registry lock, so nobody can open the file
    // between the flush and the state going away.
    static void release(std::shared_ptr<FileState>& state) {
        std::lock_guard<std::mutex> lock{registry_mutex()};
        if (state.use_count() == 1) {
            try {
                std::lock_guard<std::mutex> state_lock{state->m_mutex};
                state->flush_pending();
            } catch (...) {  // a destructor cannot report; close() does
            }
        }
        state.reset();
    }

    void initialize() {
        const auto size = std::filesystem::exists(m_path) ? std::filesystem::file_size(m_path) : uintmax_t{0};
        if (size == 0) {
            if (const auto parent = m_path.parent_path(); !parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            std::ofstream created{m_path, std::ios::binary | std::ios::trunc};
            OPENVINO_ASSERT(created, "Cannot create multi-blob file ", m_path);
            created.close();
            m_end = 0;
            append_shared_context({}, {});  // the mandatory first shared context, empty
        } else {
            scan(size);
        }
    }

    // Rebuilds what the file holds from its containers; anything invalid, a broken tail included, is an error.
    void scan(uint64_t size) {
        const auto mapped = load_mmap_object(m_path);
        OPENVINO_ASSERT(mapped && mapped->size() == size, "Cannot map multi-blob file ", m_path);
        const std::byte* data = mapped->data();

        for (uint64_t offset = 0; offset < size;) {
            const auto left = size - offset;
            OPENVINO_ASSERT(left >= sizeof(Header), invalid(offset, "the header is cut short"));
            const auto& header = Header::view(static_cast<const void*>(data + offset));
            OPENVINO_ASSERT(offset % multi_blob_alignment == 0, invalid(offset, "the container is not aligned"));
            OPENVINO_ASSERT(is_valid_header_fields(header) && header.container_size <= left,
                            invalid(offset, "the header is not valid"));
            OPENVINO_ASSERT(offset != 0 || header.magic == BlobMagic::multi,
                            invalid(offset, "the first container is not a shared context"));

            const ContainerView container{data + offset, static_cast<size_t>(header.container_size)};
            OPENVINO_ASSERT(container.validate(), invalid(offset, "the container is not valid"));
            if (header.magic == BlobMagic::multi) {
                read_shared_context(container, data, offset);
            }
            offset += header.container_size;
        }
        m_end = size;
    }

    std::string invalid(uint64_t offset, const char* why) const {
        return "Invalid multi-blob file " + m_path.string() + " at offset " + std::to_string(offset) + ": " + why;
    }

    void read_shared_context(const ContainerView& container, const std::byte* data, uint64_t offset) {
        const auto container_end = offset + container.size();
        for (size_t i = 0; i < container.manifest_count(); ++i) {
            const auto& entry = (&container.manifest())[i];
            if (entry.device != any_device_id || entry.tag.id() != shared_context) {
                continue;
            }
            OPENVINO_ASSERT(entry.tag.is_pointer(), invalid(offset, "the shared context is inline"));
            const auto payload = decode_shared_context(container.section(entry));
            OPENVINO_ASSERT(payload, invalid(offset, "the shared context is malformed"));
            merge(*payload, data, offset, container_end);
        }
    }

    void merge(const SharedContextPayload& payload, const std::byte* data, uint64_t offset, uint64_t container_end) {
        for (const auto& [source_id, link] : payload.sources) {
            OPENVINO_ASSERT(link >= sizeof(Header) && link < container_end, invalid(offset, "a source link is wrong"));
            OPENVINO_ASSERT(m_sources.emplace(source_id, link).second, invalid(offset, "a source is listed twice"));
        }
        for (const auto& [source_id, metas] : payload.constants) {
            OPENVINO_ASSERT(m_sources.count(source_id) != 0, invalid(offset, "constants of an unknown source"));
            for (const auto& [constant_id, meta] : metas) {
                OPENVINO_ASSERT(m_constants[source_id].emplace(constant_id, meta).second,
                                invalid(offset, "a constant is listed twice"));
            }
        }
        for (const auto& [blob_id, link] : payload.models) {
            OPENVINO_ASSERT(offset >= sizeof(Header) && link <= offset - sizeof(Header) &&
                                Header::view(static_cast<const void*>(data + link)).magic == BlobMagic::single,
                            invalid(offset, "a model link is wrong"));
            OPENVINO_ASSERT(m_models.emplace(blob_id, link).second, invalid(offset, "a model is listed twice"));
        }
    }

    using NewSources = std::map<wsh::DataID, wsh::WeightBuffer>;  // ordered, so the layout is stable

    void write_context(const wsh::Context& context) {
        NewSources new_sources;
        for (const auto& [source_id, source] : context.m_cache_sources) {
            if (m_sources.count(source_id) == 0) {
                if (auto buffer = source.m_weights.lock()) {
                    new_sources.emplace(source_id, std::move(buffer));
                }
            }
        }

        wsh::WeightRegistry new_constants;
        for (const auto& [source_id, metas] : context.m_weight_registry) {
            if (m_sources.count(source_id) == 0 && new_sources.count(source_id) == 0) {
                continue;
            }
            const auto known = m_constants.find(source_id);
            for (const auto& [constant_id, meta] : metas) {
                if (known == m_constants.end() || known->second.count(constant_id) == 0) {
                    new_constants[source_id].emplace(constant_id, meta);
                }
            }
        }

        if (new_sources.empty() && new_constants.empty() && m_pending_models.empty()) {
            return;
        }
        append_shared_context(std::move(new_sources), std::move(new_constants));
    }

    void write_model(uint64_t blob_id, const ICacheManager::StreamWriter& writer) {
        OPENVINO_ASSERT(m_models.count(blob_id) == 0 && m_pending_models.count(blob_id) == 0,
                        "Model ",
                        blob_id,
                        " is already in ",
                        m_path);
        ScopedLocale c_locale(LC_ALL, "C");
        const auto start = append(BlobMagic::single, [&writer](std::fstream& stream, uint64_t) {
            writer(stream);
        });
        m_pending_models.emplace(blob_id, start);
    }

    void flush_pending() {
        if (!m_pending_models.empty()) {
            append_shared_context({}, {});
        }
    }

    // Appends one shared-context container holding the new sources and constants plus every pending model record.
    void append_shared_context(NewSources new_sources, wsh::WeightRegistry new_constants) {
        const auto start = align_up(m_end);

        SharedContextPayload payload;
        payload.constants = std::move(new_constants);
        payload.models = m_pending_models;
        std::vector<OffsetType> promised;
        uint64_t cursor = sizeof(Header);
        for (const auto& [source_id, buffer] : new_sources) {
            cursor = (cursor + multi_blob_alignment - 1) / multi_blob_alignment * multi_blob_alignment;
            promised.push_back(start + cursor);
            payload.sources.emplace(source_id, start + cursor);
            cursor += buffer->size();
        }

        append(BlobMagic::multi, [&](std::fstream& stream, uint64_t container_start) {
            OPENVINO_ASSERT(container_start == start, "The shared-context container moved");
            auto writer = DeferredWriter::open(stream, BlobMagic::multi);
            OPENVINO_ASSERT(writer, "Cannot write to ", m_path);
            for (const auto& [source_id, buffer] : new_sources) {
                writer->add_section(
                    any_device_id,
                    weights_tag,
                    util::MemoryView{reinterpret_cast<const std::byte*>(buffer->get_ptr()), buffer->size()},
                    SectionAlignment{multi_blob_alignment, 1});
            }
            const auto encoder = std::make_shared<SharedContextEncoder>(payload);
            if (encoder->size() != 0) {
                writer->add_section(any_device_id, shared_context_tag, encoder->size(), encoder);
            }
            const auto written = writer->finalize();
            OPENVINO_ASSERT(!written, "Cannot write the shared context to ", m_path, ": ", written.message());
            verify_weights_offsets(stream, container_start, promised);
        });

        for (const auto& [source_id, link] : payload.sources) {
            m_sources.emplace(source_id, link);
        }
        for (const auto& [source_id, metas] : payload.constants) {
            m_constants[source_id].insert(metas.begin(), metas.end());
        }
        m_models.insert(m_pending_models.begin(), m_pending_models.end());
        m_pending_models.clear();
    }

    // Places one container at the next aligned position: `write` produces it, then it is padded to the next
    // boundary. Returns where it starts. On any failure the file is cut back.
    template <class Write>
    uint64_t append(BlobMagic magic, Write&& write) {
        OPENVINO_ASSERT(std::filesystem::file_size(m_path) == m_end, "File ", m_path, " was changed by someone else");
        const auto start = align_up(m_end);
        uint64_t new_end = 0;
        try {
            std::fstream stream{m_path, std::ios::binary | std::ios::in | std::ios::out};
            OPENVINO_ASSERT(stream.is_open(), "Cannot open ", m_path, " for writing");
            stream.seekp(static_cast<std::streamoff>(m_end));
            write_zeros(stream, start - m_end);
            OPENVINO_ASSERT(stream, "Cannot write to ", m_path);

            write(stream, start);
            new_end = finish(stream, start, magic);
            stream.flush();
            OPENVINO_ASSERT(stream, "Cannot write to ", m_path);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::resize_file(m_path, m_end, ignored);
            throw;
        }
        m_end = new_end;
        return start;
    }

    // Checks that exactly one container of the expected kind was written, then pads it up to the boundary
    // and counts the padding in its size so that readers stepping by container_size land on the next one.
    uint64_t finish(std::fstream& stream, uint64_t start, BlobMagic magic) {
        const auto written_end = static_cast<uint64_t>(stream.tellp());
        OPENVINO_ASSERT(stream && written_end >= start + sizeof(Header), "Nothing was written to ", m_path);

        Header header{};
        stream.seekg(static_cast<std::streamoff>(start));
        stream.read(reinterpret_cast<char*>(&header), sizeof(header));
        OPENVINO_ASSERT(stream && is_valid_header_fields(header) && header.magic == magic &&
                            header.container_size == written_end - start,
                        "What was written to ",
                        m_path,
                        " is not exactly one container of the expected kind");

        const auto padded_end = align_up(written_end);
        stream.seekp(static_cast<std::streamoff>(written_end));
        write_zeros(stream, padded_end - written_end);

        const SizeType padded_size = padded_end - start;
        stream.seekp(static_cast<std::streamoff>(start + offsetof(Header, container_size)));
        stream.write(reinterpret_cast<const char*>(&padded_size), sizeof(padded_size));
        return padded_end;
    }
};

MultiBlobWriter::MultiBlobWriter(std::filesystem::path path) : m_file_state{FileState::acquire(path)} {}

MultiBlobWriter::~MultiBlobWriter() {
    FileState::release(m_file_state);
}

void MultiBlobWriter::write_context(const wsh::Context& context) {
    std::lock_guard<std::mutex> lock{m_file_state->m_mutex};
    m_file_state->write_context(context);
}

void MultiBlobWriter::write_model(uint64_t blob_id, const ICacheManager::StreamWriter& writer) {
    std::lock_guard<std::mutex> lock{m_file_state->m_mutex};
    m_file_state->write_model(blob_id, writer);
}

void MultiBlobWriter::register_external_source(wsh::DataID, util::MemoryView) {}

void MultiBlobWriter::close() {
    std::lock_guard<std::mutex> lock{m_file_state->m_mutex};
    m_file_state->flush_pending();
}

}  // namespace v1
}  // namespace ov::runtime::hsm
