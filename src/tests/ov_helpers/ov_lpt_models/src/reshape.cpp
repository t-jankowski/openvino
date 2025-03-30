// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/reshape.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> ReshapeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    std::shared_ptr<Node> reshape_pattern;
    if (!reshapeConstValues.empty()) {
        reshape_pattern =
            ov::op::v0::Constant::create(ov::element::i64, Shape{reshapeConstValues.size()}, reshapeConstValues);
    } else {
        reshape_pattern = std::make_shared<ov::op::v0::ShapeOf>(dequantizationOp);
    }

    const auto reshape = std::make_shared<ov::op::v1::Reshape>(dequantizationOp, reshape_pattern, true);
    reshape->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(reshape) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReshapeFunction");
}

std::shared_ptr<ov::Model> ReshapeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ov::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const std::shared_ptr<Node> reshape = std::make_shared<ov::op::v1::Reshape>(
        quantizationOp,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ reshapeConstValues.size() }, reshapeConstValues),
        true);

    ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(reshape) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReshapeFunction");
}

std::shared_ptr<ov::Model> ReshapeFunction::getReference(
    const ov::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<Node> reshape_pattern;
    if (!reshapeConstValues.empty()) {
        reshape_pattern =
            ov::op::v0::Constant::create(ov::element::i64, Shape{reshapeConstValues.size()}, reshapeConstValues);
    } else {
        reshape_pattern = makeDequantization(quantizationOpBefore, dequantizationAfter);
        reshape_pattern = std::make_shared<ov::op::v0::ShapeOf>(reshape_pattern);
    }

    const auto reshape = std::make_shared<ov::op::v1::Reshape>(quantizationOpBefore, reshape_pattern, true);
    if (quantizationOpBefore->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*quantizationOpBefore) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }
    if (reshape->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*reshape) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(reshape, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReshapeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov


