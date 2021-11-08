// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset7.hpp>

#include "op_table.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {
OutputVector translate_softmax_op(const NodeContext& node) {
    auto ng_inp = node.get_input(0);
    // todo: switch to opset8::Softmax when is ready and delete Dynamic rank limitation
    TF_OP_VALIDATION_CHECK(node, ng_inp.get_partial_shape().rank().is_static(), "Dynamic rank is not supported.");
    size_t axis = ng_inp.get_partial_shape().rank().get_length() - 1;
    auto res = make_shared<opset7::Softmax>(ng_inp, axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov