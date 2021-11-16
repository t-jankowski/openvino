// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/frontend_manager.hpp"

#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/place.hpp"
#include "ngraph/except.hpp"
#include "openvino/util/env_util.hpp"
#include "plugin_loader.hpp"
#include "so_extension.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

//----------- FrontEndManager ---------------------------
class FrontEndManager::Impl {
    std::vector<PluginHandle> m_loaded_libs;  // must be a first class member (destroyed last)
    std::map<std::string, FrontEndFactory> m_factories;

public:
    Impl() {
        register_plugins();
    }

    ~Impl() = default;

    FrontEnd::Ptr load_by_framework(const std::string& framework) {
        FRONT_END_INITIALIZATION_CHECK(m_factories.count(framework),
                                       "FrontEnd for Framework ",
                                       framework,
                                       " is not found");
        return m_factories[framework]();
    }

    std::vector<std::string> available_front_ends() const {
        std::vector<std::string> keys;

        std::transform(m_factories.begin(),
                       m_factories.end(),
                       std::back_inserter(keys),
                       [](const std::pair<std::string, FrontEndFactory>& item) {
                           return item.first;
                       });
        return keys;
    }

    FrontEnd::Ptr load_by_model(const std::vector<std::shared_ptr<Variant>>& variants) {
        for (const auto& factory : m_factories) {
            auto FE = factory.second();
            if (FE->supported(variants)) {
                return FE;
            }
        }
        return FrontEnd::Ptr();
    }

    void register_front_end(const std::string& name, FrontEndFactory creator) {
        m_factories.insert({name, creator});
    }

private:
    void register_plugins() {
        auto register_from_dir = [&](const std::string& dir) {
            if (!dir.empty()) {
                auto plugins = load_plugins(dir);
                for (auto& plugin : plugins) {
                    register_front_end(plugin.m_plugin_info.m_name, plugin.m_plugin_info.m_creator);
                    m_loaded_libs.push_back(std::move(plugin.m_lib_handle));
                }
            }
        };
        std::string env_path = ov::util::getenv_string("OV_FRONTEND_PATH");
        if (!env_path.empty()) {
            auto start = 0u;
            auto sep_pos = env_path.find(PathSeparator, start);
            while (sep_pos != std::string::npos) {
                register_from_dir(env_path.substr(start, sep_pos - start));
                start = sep_pos + 1;
                sep_pos = env_path.find(PathSeparator, start);
            }
            register_from_dir(env_path.substr(start, sep_pos));
        } else {
            register_from_dir(get_frontend_library_path());
        }
    }
};

FrontEndManager::FrontEndManager() : m_impl(new Impl()) {}

FrontEndManager::FrontEndManager(FrontEndManager&&) = default;
FrontEndManager& FrontEndManager::operator=(FrontEndManager&&) = default;

FrontEndManager::~FrontEndManager() = default;

FrontEnd::Ptr FrontEndManager::load_by_framework(const std::string& framework) {
    return m_impl->load_by_framework(framework);
}

FrontEnd::Ptr FrontEndManager::load_by_model_impl(const std::vector<std::shared_ptr<Variant>>& variants) {
    return m_impl->load_by_model(variants);
}

std::vector<std::string> FrontEndManager::get_available_front_ends() const {
    return m_impl->available_front_ends();
}

void FrontEndManager::register_front_end(const std::string& name, FrontEndFactory creator) {
    m_impl->register_front_end(name, creator);
}

template <>
FrontEnd::Ptr FrontEndManager::load_by_model(const std::vector<std::shared_ptr<Variant>>& variants) {
    return load_by_model_impl(variants);
}

//----------- FrontEnd ---------------------------

FrontEnd::FrontEnd() = default;

FrontEnd::~FrontEnd() = default;

bool FrontEnd::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<std::shared_ptr<Variant>>& params) const {
    FRONT_END_NOT_IMPLEMENTED(load_impl);
}
std::shared_ptr<ngraph::Function> FrontEnd::convert(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

void FrontEnd::convert(std::shared_ptr<ngraph::Function>) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<ngraph::Function> FrontEnd::convert_partially(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(convert_partially);
}

std::shared_ptr<ngraph::Function> FrontEnd::decode(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(std::shared_ptr<ngraph::Function> function) const {
    FRONT_END_NOT_IMPLEMENTED(normalize);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    // Each frontend can support own set of extensions, so this method should be implemented on the frontend side
}

void FrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    for (const auto& ext : extensions)
        add_extension(ext);
}

void FrontEnd::add_extension(const std::string& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void FrontEnd::add_extension(const std::wstring& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}
#endif

std::string FrontEnd::get_name() const {
    return std::string();
}

//----------- InputModel ---------------------------
std::vector<Place::Ptr> InputModel::get_inputs() const {
    return {};
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return {};
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index) {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index) {
    return nullptr;
}

void InputModel::set_name_for_tensor(Place::Ptr tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_tensor);
}

void InputModel::add_name_for_tensor(Place::Ptr tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(add_name_for_tensor);
}

void InputModel::set_name_for_operation(Place::Ptr operation, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_operation);
}

void InputModel::free_name_for_tensor(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_tensor);
}

void InputModel::free_name_for_operation(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_operation);
}

void InputModel::set_name_for_dimension(Place::Ptr place, size_t shape_dim_index, const std::string& dim_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_dimension);
}

void InputModel::cut_and_add_new_input(Place::Ptr place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_input);
}

void InputModel::cut_and_add_new_output(Place::Ptr place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_output);
}

Place::Ptr InputModel::add_output(Place::Ptr place) {
    FRONT_END_NOT_IMPLEMENTED(add_output);
}

void InputModel::remove_output(Place::Ptr place) {
    FRONT_END_NOT_IMPLEMENTED(remove_output);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(extract_subgraph);
}

// Setting tensor properties
void InputModel::set_partial_shape(Place::Ptr place, const ngraph::PartialShape&) {
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

ngraph::PartialShape InputModel::get_partial_shape(Place::Ptr place) const {
    FRONT_END_NOT_IMPLEMENTED(get_partial_shape);
}

void InputModel::set_element_type(Place::Ptr place, const ngraph::element::Type&) {
    FRONT_END_NOT_IMPLEMENTED(set_element_type);
}

void InputModel::set_tensor_value(Place::Ptr place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_value);
}

void InputModel::set_tensor_partial_value(Place::Ptr place, const void* min_value, const void* max_value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_partial_value);
}

//----------- Place ---------------------------
std::vector<std::string> Place::get_names() const {
    FRONT_END_NOT_IMPLEMENTED(get_names);
}

std::vector<Place::Ptr> Place::get_consuming_operations() const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(int output_port_index) const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName, int outputPortIndex) const {
    return {};
}

Place::Ptr Place::get_target_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation() const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name, int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port() const {
    return nullptr;
}

Place::Ptr Place::get_output_port(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name, int output_port_index) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_ports() const {
    return {};
}

bool Place::is_input() const {
    FRONT_END_NOT_IMPLEMENTED(is_input);
}

bool Place::is_output() const {
    FRONT_END_NOT_IMPLEMENTED(is_output);
}

bool Place::is_equal(Ptr another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal);
}

bool Place::is_equal_data(Ptr another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal_data);
}

Place::Ptr Place::get_source_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName, int outputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName) const {
    return {};
}
