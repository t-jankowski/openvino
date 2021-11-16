// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs pow(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto dtype = x.get_element_type();
    Output<Node> factor_node;
    if (node.has_ng_input("FactorTensor")) {
        factor_node = node.get_ng_input("FactorTensor");
        if (factor_node.get_element_type() != dtype)
            factor_node = std::make_shared<ngraph::opset6::Convert>(factor_node, dtype);
    } else {
        factor_node = ngraph::opset6::Constant::create(dtype, Shape{1}, {node.get_attribute<float>("factor")});
    }

    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Power>(x, factor_node)}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
