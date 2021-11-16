// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_frontend/frontend.hpp"

#include <array>
#include <vector>

#include "ir_frontend/model.hpp"
#include "ir_frontend/utility.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "openvino/core/variant.hpp"
#include "openvino/util/file_util.hpp"
#include "so_extension.hpp"
#include "xml_parse_utils.h"

using namespace ov;

namespace ov {
namespace frontend {
namespace {

inline size_t GetIRVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

/**
 * @brief Extracts IR version from model stream
 * @param model Models stream
 * @return IR version, 0 if model does represent IR
 */
size_t GetIRVersion(std::istream& model) {
    std::array<char, 512> header{};

    model.seekg(0, model.beg);
    model.read(header.data(), header.size());
    model.clear();
    model.seekg(0, model.beg);

    pugi::xml_document doc;
    auto res =
        doc.load_buffer(header.data(), header.size(), pugi::parse_default | pugi::parse_fragment, pugi::encoding_utf8);

    if (res == pugi::status_ok) {
        pugi::xml_node root = doc.document_element();

        std::string node_name = root.name();
        std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

        if (node_name == "net") {
            return GetIRVersion(root);
        }
    }

    return 0;
}

}  // namespace

bool FrontEndIR::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    std::ifstream local_model_stream;
    std::istream* provided_model_stream = nullptr;

    if (variants.empty() || variants.size() > 3) {
        return false;
    }

    const auto& model_variant = variants[0];
    if (ov::is_type<ov::VariantWrapper<std::string>>(model_variant)) {
        const auto& path = ov::as_type_ptr<ov::VariantWrapper<std::string>>(model_variant)->get();
        local_model_stream.open(path, std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (ov::is_type<ov::VariantWrapper<std::wstring>>(model_variant)) {
        const auto& path = ov::as_type_ptr<ov::VariantWrapper<std::wstring>>(model_variant)->get();
        local_model_stream.open(path, std::ios::in | std::ifstream::binary);
#endif
    } else if (ov::is_type<ov::VariantWrapper<std::istream*>>(model_variant)) {
        provided_model_stream = ov::as_type_ptr<ov::VariantWrapper<std::istream*>>(model_variant)->get();
    } else if (ov::is_type<ov::VariantWrapper<std::istringstream*>>(model_variant)) {
        provided_model_stream = ov::as_type_ptr<ov::VariantWrapper<std::istringstream*>>(model_variant)->get();
    }

    if (provided_model_stream && local_model_stream.is_open()) {
        return false;
    }

    size_t version;
    if (provided_model_stream) {
        version = GetIRVersion(*provided_model_stream);
    } else if (local_model_stream.is_open()) {
        version = GetIRVersion(local_model_stream);
        local_model_stream.close();
    } else {
        return false;
    }

    return version >= 10 && version <= 11;
}

void FrontEndIR::add_extension(const ov::Extension::Ptr& ext) {
    if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
        if (std::dynamic_pointer_cast<ov::BaseOpExtension>(so_ext->extension())) {
            shared_objects.emplace_back(so_ext->shared_object());
            extensions.emplace_back(so_ext->extension());
        }
    }
    if (std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
        extensions.emplace_back(ext);
}

InputModel::Ptr FrontEndIR::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    std::ifstream local_model_stream;
    std::istream* provided_model_stream = nullptr;
    std::shared_ptr<ngraph::runtime::AlignedBuffer> weights;

    auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> exts;
        for (const auto& ext : extensions) {
            if (auto base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
                exts.insert({base_ext->get_type_info(), base_ext});
        }
        return exts;
    };

    auto create_input_model = [&]() -> std::shared_ptr<InputModelIR> {
        if (provided_model_stream) {
            return std::make_shared<InputModelIR>(*provided_model_stream, weights, create_extensions_map());
        } else if (local_model_stream.is_open()) {
            auto input_model = std::make_shared<InputModelIR>(local_model_stream, weights, create_extensions_map());
            local_model_stream.close();
            return input_model;
        }
        return nullptr;
    };

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring weights_path, model_path;
#else
    std::string weights_path, model_path;
#endif

    // Load model
    const auto& model_variant = variants.at(0);
    if (ov::is_type<ov::VariantWrapper<std::string>>(model_variant)) {
        const auto& tmp_path = ov::as_type_ptr<ov::VariantWrapper<std::string>>(model_variant)->get();
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        model_path = ov::util::string_to_wstring(tmp_path.c_str());
#else
        model_path = tmp_path;
#endif
        local_model_stream.open(model_path, std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (ov::is_type<ov::VariantWrapper<std::wstring>>(model_variant)) {
        model_path = ov::as_type_ptr<ov::VariantWrapper<std::wstring>>(model_variant)->get();
        local_model_stream.open(model_path, std::ios::in | std::ifstream::binary);
#endif
    } else if (ov::is_type<ov::VariantWrapper<std::istream*>>(model_variant)) {
        provided_model_stream = ov::as_type_ptr<ov::VariantWrapper<std::istream*>>(model_variant)->get();
    } else if (ov::is_type<ov::VariantWrapper<std::istringstream*>>(model_variant)) {
        provided_model_stream = ov::as_type_ptr<ov::VariantWrapper<std::istringstream*>>(model_variant)->get();
    }

    // Check weights and extensions
    for (size_t variant_id = 1; variant_id < variants.size(); ++variant_id) {
        const auto& variant = variants.at(variant_id);
        if (ov::is_type<ov::VariantWrapper<std::string>>(variant)) {
            const auto& tmp_path = ov::as_type_ptr<ov::VariantWrapper<std::string>>(variant)->get();
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            weights_path = ov::util::string_to_wstring(tmp_path.c_str());
#else
            weights_path = tmp_path;
#endif
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        } else if (ov::is_type<ov::VariantWrapper<std::wstring>>(variant)) {
            weights_path = ov::as_type_ptr<ov::VariantWrapper<std::wstring>>(variant)->get();
#endif
        } else if (ov::is_type<VariantWrapper<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(variant)) {
            weights = ov::as_type_ptr<VariantWrapper<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(variant)->get();
        }
    }

    // Find weights if only path to xml was provided
    if (weights_path.empty()) {
        auto pos = model_path.rfind('.');
        if (pos != model_path.npos)
            weights_path = model_path.substr(0, pos);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        weights_path += L".bin";
#else
        weights_path += ".bin";
#endif
        if (!FileUtils::fileExist(weights_path)) {
            weights_path.clear();
        }
    }

    if (!weights_path.empty()) {
        std::ifstream bin_stream;
        bin_stream.open(weights_path, std::ios::binary);
        if (!bin_stream.is_open())
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            IR_THROW("Weights file " + ov::util::wstring_to_string(weights_path) + " cannot be opened!");
#else
            IR_THROW("Weights file " + weights_path + " cannot be opened!");
#endif

        bin_stream.seekg(0, std::ios::end);
        size_t file_size = bin_stream.tellg();
        bin_stream.seekg(0, std::ios::beg);

        auto aligned_weights_buffer = std::make_shared<ngraph::runtime::AlignedBuffer>(file_size);
        bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
        bin_stream.close();

        weights = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
            aligned_weights_buffer->get_ptr<char>(),
            aligned_weights_buffer->size(),
            aligned_weights_buffer);
    }

    return create_input_model();
}

std::shared_ptr<ov::Function> FrontEndIR::convert(InputModel::Ptr model) const {
    auto ir_model = std::dynamic_pointer_cast<InputModelIR>(model);
    OPENVINO_ASSERT(ir_model != nullptr);
    return ir_model->convert();
}

std::string FrontEndIR::get_name() const {
    return "ir";
}
}  // namespace frontend
}  // namespace ov

extern "C" IR_API ov::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" IR_API void* GetFrontEndData() {
    frontend::FrontEndPluginInfo* res = new frontend::FrontEndPluginInfo();
    res->m_name = "ir";
    res->m_creator = []() {
        return std::make_shared<frontend::FrontEndIR>();
    };
    return res;
}
