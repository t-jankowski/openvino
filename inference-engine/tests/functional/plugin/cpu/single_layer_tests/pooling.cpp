// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using poolLayerCpuTestParamsSet = std::tuple<LayerTestsDefinitions::poolSpecificParams,
                                             InputShape,
                                             ElementType,
                                             bool,
                                             CPUSpecificParams,
                                             fusingSpecificParams>;

class PoolingLayerCPUTest : public testing::WithParamInterface<poolLayerCpuTestParamsSet>,
                            virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
        LayerTestsDefinitions::poolSpecificParams basicParamsSet;
        InputShape inputShapes;
        ElementType inPrc;
        bool isInt8;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams) = obj.param;

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        bool excludePad;
        std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

        std::ostringstream results;
        results << "IS=(";
        results << CommonTestUtils::partialShape2str({inputShapes.first}) << ")_";
        results << "TS=";
        for (const auto& shape : inputShapes.second) {
            results << CommonTestUtils::vec2str(shape) << "_";
        }
        results << "Prc=" << inPrc << "_";
        switch (poolType) {
            case ngraph::helpers::PoolingTypes::MAX:
                results << "MaxPool_";
                break;
            case ngraph::helpers::PoolingTypes::AVG:
                results << "AvgPool_";
                results << "ExcludePad=" << excludePad << "_";
                break;
        }
        results << "K" << CommonTestUtils::vec2str(kernel) << "_";
        results << "S" << CommonTestUtils::vec2str(stride) << "_";
        results << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        results << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        results << "Rounding=" << roundingType << "_";
        results << "AutoPad=" << padType << "_";
        results << "INT8=" << isInt8 << "_";

        results << CPUTestsBase::getTestCaseName(cpuParams);
        results << CpuTestWithFusing::getTestCaseName(fusingParams);
        return results.str();

        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        LayerTestsDefinitions::poolSpecificParams basicParamsSet;
        InputShape inputShapes;
        ElementType inPrc;
        bool isInt8;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams) = this->GetParam();

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        bool excludePad;
        std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = selectedType + "_" + InferenceEngine::details::convertPrecision(inPrc).name();

        init_input_shapes({inputShapes});

        auto params = ngraph::builder::makeDynamicParams(inPrc, inputDynamicShapes);

        std::shared_ptr<ngraph::Node> poolInput = params[0];
        if (isInt8) {
            ov::Shape newShape(poolInput->get_output_partial_shape(0).size(), 1);
            poolInput = ngraph::builder::makeFakeQuantize(poolInput, inPrc, 256, newShape);
        }

        std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(poolInput,
                                                                             stride,
                                                                             padBegin,
                                                                             padEnd,
                                                                             kernel,
                                                                             roundingType,
                                                                             padType,
                                                                             excludePad,
                                                                             poolType);

        function = makeNgraphFunction(inPrc, params, pooling, "PoolingCPU");
    }
};

TEST_P(PoolingLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
//     CheckPluginRelatedResults(executableNetwork, "Pooling");
}

namespace {

const auto avx512 = CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
const auto avx = CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"};
const auto sse42 = CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};
const auto ref = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<CPUSpecificParams> vecCpuConfigs = {ref, sse42, avx, avx512};
const std::vector<ElementType> inpOutPrecision = {ElementType::f32/*, ElementType::bf16*/};

const std::vector<InputShape> inputShapes4D = {
        { {}, {{3, 4, 64, 64}} },
        { {}, {{2, 8, 8, 12}} },
        { {}, {{1, 16, 16, 12}} },
        { {}, {{1, 21, 8, 4}} },
        { {}, {{1, 32, 8, 8}} },
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 32, 8, 8},
                {1, 21, 8, 4},
                {2, 8, 8, 12}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}},
            // target
            {
                {3, 4, 64, 64},
                {1, 16, 16, 12},
                {1, 32, 8, 8}
            }
        }
};

const std::vector<InputShape> inputShapes5D = {
        { {}, {{1, 4, 16, 16, 16}} },
        { {}, {{2, 8, 8, 8, 8}} },
        { {}, {{2, 16, 12, 16, 20}} },
        { {}, {{1, 19, 16, 20, 8}} },
        { {}, {{1, 32, 16, 8, 12}} },
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {2, 8, 8, 8, 8},
                {1, 19, 16, 20, 8},
                {1, 4, 16, 16, 16}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}, {1, 25}},
            // target
            {
                {1, 4, 16, 16, 16},
                {1, 32, 16, 8, 12},
                {3, 16, 4, 8, 3}
            }
        }
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                            ::testing::Combine(
                            ::testing::ValuesIn(paramsMax4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_RefOnly),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::Values(ref),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 3, 4}, {2, 2, 2}, {1, 1, 1}, {1, 2, 3},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(paramsMax5D),
                             ::testing::ValuesIn(inputShapes5D),
                             ::testing::ValuesIn(inpOutPrecision),
                             ::testing::Values(false),
                             ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                             ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision),
                              ::testing::Values(false),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D_RefOnly),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision),
                              ::testing::Values(false),
                              ::testing::Values(ref),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);

/* === Fusing === */

const auto avx512_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"};
const auto avx512_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx512"}, "jit_avx512"};

const auto avx2_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"};
const auto avx2_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx2"}, "jit_avx2"};

const auto sse42_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"};
const auto sse42_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_sse42"}, "jit_sse42"};

const std::vector<CPUSpecificParams> vecCpuConfigsFusing_4D = {sse42_nhwc, avx2_nhwc, avx512_nhwc};
const std::vector<CPUSpecificParams> vecCpuConfigsFusing_5D = {sse42_ndhwc, avx2_ndhwc, avx512_ndhwc};

std::vector<fusingSpecificParams> fusingParamsSet {
    emptyFusingSpec,
    fusingFakeQuantizePerTensor,
    fusingFakeQuantizePerChannel,
};

const std::vector<InputShape> inputShapes4D_int8 = {
        { {}, {{3, 4, 64, 64}} },
        { {}, {{2, 8, 8, 12}} },
        { {}, {{1, 16, 16, 12}} },
        { {}, {{1, 21, 8, 4}} },
        { {}, {{1, 32, 8, 8}} },
        {
            // dynamic
            {-1, 32, -1, -1},
            // target
            {
                {1, 32, 8, 8},
                {1, 32, 8, 4},
                {2, 32, 8, 12}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 64}, {1, 64}},
            // target
            {
                {3, 16, 32, 32},
                {1, 16, 16, 12},
                {1, 16, 8, 8}
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_I8, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg4D),
                              ::testing::ValuesIn(inputShapes4D_int8),
                              ::testing::Values(ElementType::f32),
                              ::testing::Values(true),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_4D)),
                              ::testing::ValuesIn(fusingParamsSet)),
                          PoolingLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_int8 = {
        { {}, {{1, 4, 16, 16, 16}} },
        { {}, {{2, 8, 8, 8, 8}} },
        { {}, {{2, 16, 12, 16, 20}} },
        { {}, {{1, 19, 16, 20, 8}} },
        { {}, {{1, 32, 16, 8, 12}} },
        {
            // dynamic
            {-1, 32, -1, -1, -1},
            // target
            {
                {2, 32, 8, 8, 8},
                {1, 32, 16, 20, 8},
                {1, 32, 16, 16, 16}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 64}, {1, 64}, {1, 25}},
            // target
            {
                {1, 16, 16, 16, 16},
                {1, 16, 16, 8, 12},
                {2, 16, 8, 8, 8}
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_I8, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D_int8),
                              ::testing::Values(ElementType::f32),
                              ::testing::Values(true),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_5D)),
                              ::testing::ValuesIn(fusingParamsSet)),
                          PoolingLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
