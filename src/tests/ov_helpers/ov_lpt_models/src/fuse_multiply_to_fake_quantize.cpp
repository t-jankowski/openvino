// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fuse_multiply_to_fake_quantize.hpp"

#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FuseMultiplyToFakeQuantizeFunction::get(
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);

    const auto constantPrecision = fqOnData.constantPrecision != ov::element::dynamic ? fqOnData.constantPrecision : ov::element::f32;
    const auto fakeQuantize = makeFakeQuantize(input, constantPrecision, fqOnData);
    const auto lastDequantization = makeDequantization(fakeQuantize, dequantization);
    lastDequantization->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(lastDequantization) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseSubtractToFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov


