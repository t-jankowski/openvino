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
NamedOutputs log(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Log>(x)}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
