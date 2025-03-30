// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/broadcast.hpp"


#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace builder {
namespace subgraph {

namespace {
template <typename T>
std::shared_ptr<ov::Node> make_broadcast(const std::shared_ptr<ov::Node>& parent, const Shape& tagetShape, const Shape& axesMapping) {
    return std::make_shared<T>(
        parent,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{ tagetShape.size() }, tagetShape),
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{ axesMapping.size() }, axesMapping));
}
} // namespace

std::shared_ptr<ov::Model> BroadcastFunction::get(
    const bool v1,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const Shape& tagetShape,
    const Shape& axesMapping,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ov::Node> parent = input;

    if (!dequantizationBefore.empty()) {
        parent = makeDequantization(parent, dequantizationBefore);
    }

    parent = v1 ?
        make_broadcast<ov::op::v1::Broadcast>(parent, tagetShape, axesMapping) :
        make_broadcast<ov::op::v3::Broadcast>(parent, tagetShape, axesMapping);
    parent->set_friendly_name("broadcast");

    if (!dequantizationAfter.empty()) {
        parent = makeDequantization(parent, dequantizationAfter);
    }

    const std::shared_ptr<ov::op::v0::Result> result = std::make_shared<ov::op::v0::Result>(parent);

    const std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "BroadcastTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov


