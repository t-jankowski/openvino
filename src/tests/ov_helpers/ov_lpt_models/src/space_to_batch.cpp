// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/space_to_batch.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/space_to_batch.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> SpaceToBatchFunction::get(const ov::PartialShape& input_shape,
                                                            const ov::element::Type input_type,
                                                            const FakeQuantizeOnData& fq_on_data,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& pads_begin,
                                                            const std::vector<size_t>& pads_end) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = fq_on_data.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, input_type, fq_on_data);

    parent = std::make_shared<ov::op::v1::SpaceToBatch>(
        parent,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ pads_begin.size() }, pads_begin),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ pads_end.size() }, pads_end));

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(parent)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SpaceToBatchFunction");
}

std::shared_ptr<ov::Model> SpaceToBatchFunction::get(const ov::PartialShape& input_shape,
                                                            const ov::element::Type input_type,
                                                            const ov::builder::subgraph::DequantizationOperations& dequantization_before,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& pads_begin,
                                                            const std::vector<size_t>& pads_end,
                                                            const ov::builder::subgraph::DequantizationOperations& dequantization_after) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = dequantization_before.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeDequantization(input, dequantization_before);

    parent = std::make_shared<ov::op::v1::SpaceToBatch>(
        parent,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ pads_begin.size() }, pads_begin),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ pads_end.size() }, pads_end));

    parent = makeDequantization(parent, dequantization_after);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(parent)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SpaceToBatchFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov


