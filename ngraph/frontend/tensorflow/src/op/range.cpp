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

OutputVector translate_range_op(const NodeContext& node) {
    auto start = node.get_input(0);
    auto stop = node.get_input(1);
    auto step = node.get_input(2);
    auto out_type = node.get_attribute<ov::element::Type>("Tidx");

    auto res = make_shared<Range>(start, stop, step, out_type);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
