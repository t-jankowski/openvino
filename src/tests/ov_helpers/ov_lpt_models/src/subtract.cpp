// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/subtract.hpp"
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {
    std::shared_ptr<ov::Model> SubtractFunction::getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape) {
        const float k = 50.f;

        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        const auto fakeQuantizeOnActivations = ov::test::utils::make_fake_quantize(
            input, precision, 256ul, { 1ul },
            { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });

        const size_t channelsValue = inputShape[1].get_length();
        const auto weights = ov::op::v0::Constant::create(
            precision,
            ov::Shape{ channelsValue, channelsValue, 1, 1 },
            std::vector<float>(channelsValue * channelsValue, 1));

        const auto convolution = std::make_shared<ov::op::v1::Convolution>(
            fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
            ov::test::utils::make_fake_quantize(weights, precision, 256ul, { 1ul }, { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k }),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(convolution) };
        std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
            results,
            ov::ParameterVector{ input },
            "SubtractTransformation");

        return function;
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ov


