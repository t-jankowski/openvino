// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <editor.hpp>
#include <frontend_manager/place.hpp>
#include <memory>
#include <sstream>

namespace ov {
namespace frontend {
class PlaceInputEdgeONNX : public Place {
public:
    PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);
    PlaceInputEdgeONNX(onnx_editor::InputEdge&& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    // internal usage
    onnx_editor::InputEdge get_input_edge() const;

    // external usage
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(Place::Ptr another) const override;
    bool is_equal_data(Place::Ptr another) const override;
    Place::Ptr get_source_tensor() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    Place::Ptr get_producing_operation() const override;
    Place::Ptr get_producing_port() const override;

private:
    onnx_editor::InputEdge m_edge;
    const std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

class PlaceOutputEdgeONNX : public Place {
public:
    PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);
    PlaceOutputEdgeONNX(onnx_editor::OutputEdge&& edge, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    // internal usage
    onnx_editor::OutputEdge get_output_edge() const;

    // external usage
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(Place::Ptr another) const override;
    bool is_equal_data(Place::Ptr another) const override;
    Place::Ptr get_target_tensor() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Place::Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;

private:
    onnx_editor::OutputEdge m_edge;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

class PlaceTensorONNX : public Place {
public:
    PlaceTensorONNX(const std::string& name, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);
    PlaceTensorONNX(std::string&& name, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);

    // external usage
    std::vector<std::string> get_names() const override;
    Place::Ptr get_producing_port() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Place::Ptr get_producing_operation() const override;
    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(Place::Ptr another) const override;
    bool is_equal_data(Place::Ptr another) const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;

    void set_name(const std::string& new_name);
    void set_name_for_dimension(size_t shape_dim_index, const std::string& dim_name);

private:
    std::string m_name;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};

class PlaceOpONNX : public Place {
public:
    PlaceOpONNX(const onnx_editor::EditorNode& node, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);
    PlaceOpONNX(onnx_editor::EditorNode&& node, std::shared_ptr<onnx_editor::ONNXModelEditor> editor);
    std::vector<std::string> get_names() const override;

    // internal usage
    const onnx_editor::EditorNode& get_editor_node() const;
    void set_name(const std::string& new_name);

    // external usage
    Place::Ptr get_output_port() const override;
    Place::Ptr get_output_port(int output_port_index) const override;
    Place::Ptr get_output_port(const std::string& output_port_name) const override;

    Place::Ptr get_input_port() const override;
    Place::Ptr get_input_port(int input_port_index) const override;
    Place::Ptr get_input_port(const std::string& input_name) const override;

    std::vector<Place::Ptr> get_consuming_ports() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    std::vector<Place::Ptr> get_consuming_operations(int output_port_index) const override;
    std::vector<Place::Ptr> get_consuming_operations(const std::string& output_port_name) const override;

    Place::Ptr get_producing_operation() const override;
    Place::Ptr get_producing_operation(int input_port_index) const override;
    Place::Ptr get_producing_operation(const std::string& input_port_name) const override;

    Place::Place::Ptr get_target_tensor() const override;
    Place::Ptr get_target_tensor(int output_port_index) const override;
    Place::Ptr get_target_tensor(const std::string& output_name) const override;

    Place::Place::Ptr get_source_tensor() const override;
    Place::Ptr get_source_tensor(int input_port_index) const override;
    Place::Ptr get_source_tensor(const std::string& input_name) const override;

    bool is_equal(Place::Ptr another) const override;
    bool is_input() const override;
    bool is_output() const override;

private:
    onnx_editor::EditorNode m_node;
    std::shared_ptr<onnx_editor::ONNXModelEditor> m_editor;
};
}  // namespace frontend
}  // namespace ov
