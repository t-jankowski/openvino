// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>
#include <openvino/opsets/opset6.hpp>
namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs assign_value(const NodeContext& node) {
    std::vector<int32_t> shape = node.get_attribute<std::vector<int32_t>>("shape");
    auto dtype = node.get_attribute<ngraph::element::Type>("dtype");
    std::shared_ptr<Node> const_node;

    switch (dtype) {
    case element::i32: {
        auto values = node.get_attribute<std::vector<int32_t>>("int32_values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    case element::f32: {
        std::vector<float> values = node.get_attribute<std::vector<float>>("fp32_values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    case element::boolean: {
        auto values = node.get_attribute<std::vector<int32_t>>("bool_values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    case element::i64: {
        auto values = node.get_attribute<std::vector<int64_t>>("int64_values");
        const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
        break;
    }
    default: {
        PDPD_OP_VALIDATION_CHECK(node, false, "assign_value only supports int32, int64, float32, bool");
        break;
    }
    }

    return node.default_single_output_mapping({const_node}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
