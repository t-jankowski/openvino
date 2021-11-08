// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <tensorflow_frontend/decoder.hpp>

#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {
namespace tf {

class TFFrameworkNode : public ::ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("TFFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    TFFrameworkNode(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs, size_t num_outputs)
        : FrameworkNode(inputs, std::max(num_outputs, size_t(1))),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name(m_decoder->get_op_type());
        set_attrs(attrs);

        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<TFFrameworkNode>(m_decoder, inputs, get_output_size());
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

private:
    std::shared_ptr<DecoderBase> m_decoder;
};
}  // namespace tf
}  // namespace frontend
}  // namespace ov
