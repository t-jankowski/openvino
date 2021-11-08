// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector translate_fill_op(const NodeContext& node) {
    auto ng_dims = node.get_input(0);
    auto ng_value = node.get_input(1);
    auto res = make_shared<Broadcast>(ng_value, ng_dims);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
