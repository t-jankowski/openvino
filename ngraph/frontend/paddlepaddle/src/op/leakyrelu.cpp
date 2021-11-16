// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs leaky_relu(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto alpha = ngraph::opset6::Constant::create(ngraph::element::f32, {1}, {node.get_attribute<float>("alpha")});
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::PRelu>(data, alpha)}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
