// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "util/test_tools.hpp"

using namespace ov;
using namespace ov::preprocess;

static std::shared_ptr<Function> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto op = std::make_shared<op::v0::Relu>(data1);
    op->set_friendly_name("Relu");
    op->get_output_tensor(0).set_names({"tensor_Relu"});
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    return std::make_shared<Function>(ResultVector{res}, ParameterVector{data1});
}

template <int N>
static std::shared_ptr<Function> create_n_inputs(element::Type type, const PartialShape& shape) {
    ResultVector res;
    ParameterVector params;
    for (size_t i = 0; i < N; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        auto op1 = std::make_shared<op::v0::Relu>(data1);
        op1->set_friendly_name("Relu" + index_str);
        auto res1 = std::make_shared<op::v0::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    return std::make_shared<Function>(res, params);
}

TEST(pre_post_process, simple_mean_scale) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().mean(1.f).scale(2.f))).build();
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, simple_mean_scale_getters) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input("tensor_input1").preprocess().mean(1).scale(2);
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, convert_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::i16))
                       .preprocess(PreProcessSteps()
                                       .convert_element_type(element::f32)
                                       .scale(2.f)
                                       .convert_element_type(element::i8)))
            .build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i16);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, convert_element_type_implicit) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    f = PrePostProcessor(f).input(InputInfo().tensor(InputTensorInfo().set_element_type(element::f32))).build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::i32);
}

TEST(pre_post_process, convert_element_type_same) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    auto old_size = f->get_ops().size();
    f = PrePostProcessor(f)
            .input(InputInfo("tensor_input1")
                       .tensor(InputTensorInfo().set_element_type(element::i32))
                       .preprocess(PreProcessSteps().convert_element_type(element::i32)))
            .build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(old_size, f->get_ops().size());
}

TEST(pre_post_process, convert_element_type_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto type_custom1 = element::Type();
    auto type_custom2 = element::Type();
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::i32))
                       .preprocess(PreProcessSteps()
                                       .custom([&type_custom1](const Output<Node>& node) {
                                           type_custom1 = node.get_element_type();
                                           return node;
                                       })
                                       .convert_element_type()
                                       .custom([&type_custom2](const Output<Node>& node) {
                                           type_custom2 = node.get_element_type();
                                           return node;
                                       })))
            .build();
    EXPECT_EQ(type_custom1, element::i32);
    EXPECT_EQ(type_custom2, element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::f32);
}

TEST(pre_post_process, empty_preprocess) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f).input(InputInfo().tensor(InputTensorInfo().set_element_type(element::i8))).build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i8);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, preprocess_assert_input_without_index) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto inp = InputInfo();
    EXPECT_ANY_THROW(f = PrePostProcessor(f).input(std::move(inp)).build());
    inp = InputInfo("some_non_existing_name");
    EXPECT_ANY_THROW(f = PrePostProcessor(f).input(std::move(inp)).build());
}

TEST(pre_post_process, convert_element_type_from_unknown) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    ASSERT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo().preprocess(
                    PreProcessSteps().convert_element_type(element::dynamic).convert_element_type(element::i32)))
                .build(),
        ov::AssertFailure);
}

TEST(pre_post_process, scale_not_float) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo().preprocess(PreProcessSteps().convert_element_type(element::i32).scale(2.0f)))
                .build(),
        ov::AssertFailure);
}

TEST(pre_post_process, mean_not_float) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().preprocess(PreProcessSteps().convert_element_type(element::i32).mean(2.0f)))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, tensor_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::f32))
                       .preprocess(PreProcessSteps().scale(2.0f).convert_element_type(element::i8)))
            .build();

    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), Layout());
}

TEST(pre_post_process, convert_color_nv12_rgb_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    f = PrePostProcessor(f)
            .input(
                InputInfo()
                    .tensor(InputTensorInfo()
                                .set_element_type(element::u8)
                                .set_color_format(ColorFormat::NV12_SINGLE_PLANE))
                    .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB).convert_element_type(element::f32)))
            .build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_nv12_bgr_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE))
                       .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR)))
            .build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes) {
    auto f = create_simple_function(element::f32, Shape{5, 2, 2, 3});
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"TestY", "TestUV"}))
                       .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR)))
            .build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_friendly_name(), "input1/TestY");
    EXPECT_EQ(*f->get_parameters()[0]->output(0).get_tensor().get_names().begin(), "tensor_input1/TestY");
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{5, 2, 2, 1}));

    EXPECT_EQ(f->get_parameters()[1]->get_friendly_name(), "input1/TestUV");
    EXPECT_EQ(*f->get_parameters()[1]->output(0).get_tensor().get_names().begin(), "tensor_input1/TestUV");
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{5, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_rgb_2_planes) {
    auto f = create_simple_function(element::f32, Shape{5, 2, 2, 3});
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                       .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
            .build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{5, 2, 2, 1}));
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{5, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes_u8_lvalue) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    auto input_tensor_info = InputTensorInfo();
    input_tensor_info.set_color_format(ColorFormat::NV12_TWO_PLANES);
    auto steps = PreProcessSteps();
    steps.convert_color(ColorFormat::BGR);
    f = PrePostProcessor(f)
            .input(InputInfo().tensor(std::move(input_tensor_info)).preprocess(std::move(steps)))
            .build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{1, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes_el_type) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    EXPECT_NO_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo()
                           .tensor(InputTensorInfo()
                                       .set_element_type(element::f32)
                                       .set_color_format(ColorFormat::NV12_TWO_PLANES))
                           .preprocess(
                               PreProcessSteps().convert_element_type(element::u8).convert_color(ColorFormat::BGR)))
                .build());

    ASSERT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
}

TEST(pre_post_process, convert_color_same_type) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    EXPECT_NO_THROW(f = PrePostProcessor(f)
                            .input(InputInfo()
                                       .tensor(InputTensorInfo().set_color_format(ColorFormat::RGB))
                                       .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
                            .build());

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, convert_color_unsupported) {
    // Feel free to update this test when more color conversions are supported in future
    auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE))
                                    .preprocess(PreProcessSteps().convert_color(ColorFormat::UNDEFINED)))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                                    .preprocess(PreProcessSteps().convert_color(ColorFormat::UNDEFINED)))
                         .build(),
                 ov::AssertFailure);

    auto colors = {ColorFormat::NV12_TWO_PLANES, ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ColorFormat::BGR};
    for (const auto& color : colors) {
        EXPECT_THROW(f = PrePostProcessor(f)
                             .input(InputInfo()
                                        .tensor(InputTensorInfo().set_color_format(ColorFormat::UNDEFINED))
                                        .preprocess(PreProcessSteps().convert_color(color)))
                             .build(),
                     ov::AssertFailure);

        EXPECT_THROW(f = PrePostProcessor(f)
                             .input(InputInfo()
                                        .tensor(InputTensorInfo().set_color_format(color))
                                        .preprocess(PreProcessSteps().convert_color(ColorFormat::UNDEFINED)))
                             .build(),
                     ov::AssertFailure);
    }
}

TEST(pre_post_process, convert_color_incorrect_subnames) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    EXPECT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo()
                           .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE, {"Test"}))
                           .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
                .build(),
        ov::AssertFailure);

    EXPECT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo().tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Test"})))
                .build(),
        ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().tensor(
                             InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"1", "2", "3"})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, convert_color_duplicate_subnames) {
    auto f = create_n_inputs<2>(element::f32, PartialShape{1, 2, 2, 3});
    f->get_parameters()[0]->get_output_tensor(0).set_names({"tensor_input1"});
    f->get_parameters()[1]->get_output_tensor(0).set_names({"tensor_input1/CustomUV"});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE,
                                                                               {"CustomY", "CustomUV"}))
                                    .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, convert_color_duplicate_internal_subnames_mean) {
    auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    for (int i = 0; i < 10; i++) {
        // Create preprocessing step several times (try to duplicate internal node names this way)
        EXPECT_NO_THROW(f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().mean(0.1f))).build());
        EXPECT_NO_THROW(f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().scale(1.1f))).build());
        EXPECT_NO_THROW(
            f = PrePostProcessor(f)
                    .input(InputInfo().preprocess(
                        PreProcessSteps().convert_element_type(element::u8).convert_element_type(element::f32)))
                    .build());
    }
    f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    for (int i = 0; i < 10; i++) {
        (f = PrePostProcessor(f)
                 .input(InputInfo()
                            .tensor(InputTensorInfo().set_layout("NHWC"))
                            .preprocess(PreProcessSteps().convert_layout("NCHW"))
                            .network(InputNetworkInfo().set_layout("NHWC")))
                 .build());
    }
    f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    auto p = PreProcessSteps();
    for (int i = 10; i < 20; i++) {
        p.resize(ResizeAlgorithm::RESIZE_LINEAR, i, i);
    }
    p.resize(ResizeAlgorithm::RESIZE_LINEAR);
    EXPECT_NO_THROW(f = PrePostProcessor(f)
                            .input(InputInfo()
                                       .tensor(InputTensorInfo().set_spatial_static_shape(480, 640))
                                       .preprocess(std::move(p))
                                       .network(InputNetworkInfo().set_layout("NHWC")))
                            .build());
}

TEST(pre_post_process, unsupported_network_color_format) {
    auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE)))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES)))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo()
                           .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                           .preprocess(PreProcessSteps().convert_layout("NCHW").convert_color(ColorFormat::RGB)))
                .build(),
        ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                                    .preprocess(PreProcessSteps().mean(0.1f).convert_color(ColorFormat::RGB)))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                                    .preprocess(PreProcessSteps().scale(2.1f).convert_color(ColorFormat::RGB)))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, custom_preprocessing) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 1, 1});
    f = PrePostProcessor(f)
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const Output<Node>& node) {
                return std::make_shared<op::v0::Abs>(node);
            })))
            .build();
    EXPECT_EQ(f->get_output_element_type(0), element::i32);
}

TEST(pre_post_process, test_lvalue) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    auto p = PrePostProcessor(f);
    auto p1 = std::move(p);
    p = std::move(p1);
    auto inputInfo = InputInfo();
    auto inputInfo2 = std::move(inputInfo);
    inputInfo = std::move(inputInfo2);
    {
        auto inputTensorInfo = InputTensorInfo();
        auto inputTensorInfo2 = std::move(inputTensorInfo);
        inputTensorInfo = std::move(inputTensorInfo2);
        auto& same = inputTensorInfo.set_element_type(element::f32);
        same.set_layout("?CHW");
        inputInfo.tensor(std::move(same));
    }
    {
        auto preprocessSteps = PreProcessSteps();
        auto preprocessSteps2 = std::move(preprocessSteps);
        preprocessSteps = std::move(preprocessSteps2);
        preprocessSteps.mean(1.f);
        preprocessSteps.scale(2.f);
        preprocessSteps.mean({1.f, 2.f, 3.f});
        preprocessSteps.scale({2.f, 3.f, 4.f});
        preprocessSteps.custom([](const Output<Node>& node) {
            return std::make_shared<op::v0::Abs>(node);
        });
        auto& same = preprocessSteps.convert_element_type(element::i8);
        inputInfo.preprocess(std::move(same));
    }
    p.input(std::move(inputInfo));
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "?CHW");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, test_2_inputs_basic) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 1, 1});
    { f = PrePostProcessor(f).input(InputInfo(1).preprocess(PreProcessSteps().mean(1.f).scale(2.0f))).build(); }
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    EXPECT_EQ(f->get_output_element_type(1), element::f32);
}

TEST(pre_post_process, reuse_network_layout_no_tensor_info) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    f->get_parameters().front()->set_layout("NC??");
    f = PrePostProcessor(f)
            .input(InputInfo().preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
            .build();
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
}

TEST(pre_post_process, reuse_network_layout_tensor_info) {
    auto f = create_simple_function(element::u8, PartialShape{Dimension::dynamic(), 3, 2, 1});
    f->get_parameters().front()->set_layout("NC??");
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::f32))
                       .preprocess(PreProcessSteps()
                                       .mean({1.f, 2.f, 3.f})
                                       .scale({2.f, 3.f, 4.f})
                                       .convert_element_type(element::u8)))
            .build();
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
}

TEST(pre_post_process, mean_scale_vector_tensor_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("NC??"))
                       .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
            .build();
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, mean_scale_dynamic_layout) {
    auto f = create_simple_function(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("N...C"))
                       .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
            .build();

    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "N...C");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, scale_vector_no_channels_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("N?HW"))
                                    .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, scale_vector_dim_mismatch) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NCHW"))
                                    .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f, 0.4f})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, scale_vector_channels_out_of_range) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("0123C"))
                                    .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_no_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(
        f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().mean({0.1f, 0.2f, 0.3f}))).build(),
        ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_dynamic_channels_shape) {
    auto f = create_simple_function(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    EXPECT_NO_THROW(f = PrePostProcessor(f)
                            .input(InputInfo()
                                       .tensor(InputTensorInfo().set_layout("NCHW"))
                                       .preprocess(PreProcessSteps().mean({0.1f, 0.2f, 0.3f})))
                            .build());
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

// Error cases for 'resize'
TEST(pre_post_process, resize_no_network_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NHWC"))
                                    .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_CUBIC)))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, tensor_spatial_shape_no_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NC?W").set_spatial_static_shape(480, 640))
                                    .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_CUBIC)))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NCH?").set_spatial_static_shape(480, 640))
                                    .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_CUBIC)))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, resize_no_tensor_height) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("N?WC"))
                                    .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                                    .network(InputNetworkInfo().set_layout("NHWC")))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, resize_no_tensor_width) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NH?C"))
                                    .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                                    .network(InputNetworkInfo().set_layout("NHWC")))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto name = f->get_results().front()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names = f->output().get_tensor().get_names();

    f = PrePostProcessor(f)
            .input(
                InputInfo().tensor(InputTensorInfo().set_layout("NHWC")).network(InputNetworkInfo().set_layout("NCHW")))
            .build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(name, f->get_results().front()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(tensor_names, f->output().get_tensor().get_names());
}

TEST(pre_post_process, preprocess_convert_layout_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("NHWC"))
                       .preprocess(PreProcessSteps().convert_layout())
                       .network(InputNetworkInfo().set_layout("NCHW")))
            .build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, preprocess_convert_layout_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();

    f = PrePostProcessor(f)
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("NCHW"))
                       .preprocess(PreProcessSteps().convert_layout("NCHW"))
                       .network(InputNetworkInfo().set_layout("NCHW")))
            .build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, preprocess_convert_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().convert_layout({0, 3, 1, 2}))).build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
}

TEST(pre_post_process, preprocess_convert_layout_dims_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    f = PrePostProcessor(f)
            .input(InputInfo().preprocess(PreProcessSteps().convert_layout(std::vector<uint64_t>{})))
            .build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 3, 480, 640}));
}

TEST(pre_post_process, preprocess_convert_layout_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());

    f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().convert_layout({0, 3, 1, 2}))).build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape::dynamic()));
}

TEST(pre_post_process, preprocess_convert_layout_invalid_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    EXPECT_THROW(
        f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().convert_layout({0, 3, 2, 2}))).build(),
        ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().preprocess(
                             PreProcessSteps().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_invalid_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());
    EXPECT_THROW(
        f = PrePostProcessor(f).input(InputInfo().preprocess(PreProcessSteps().convert_layout({0, 3, 2, 2}))).build(),
        ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo().preprocess(
                             PreProcessSteps().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, preprocess_reverse_channels_multiple_planes) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    EXPECT_THROW(
        f = PrePostProcessor(f)
                .input(InputInfo()
                           .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"}))
                           .preprocess(PreProcessSteps().reverse_channels()))
                .build(),
        ov::AssertFailure);
}

TEST(pre_post_process, preprocess_reverse_channels_no_c_dim) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("N?HW"))
                                    .preprocess(PreProcessSteps().reverse_channels()))
                         .build(),
                 ov::AssertFailure);
}

// --- PostProcess - set/convert element type ---

TEST(pre_post_process, postprocess_convert_element_type_explicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto name = f->output().get_node_shared_ptr()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto old_names = f->output().get_tensor().get_names();
    f = PrePostProcessor(f)
            .output(OutputInfo().postprocess(PostProcessSteps().convert_element_type(element::u8)))
            .build();
    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->output().get_tensor().get_names(), old_names);
    EXPECT_EQ(old_names.count("tensor_output1"), 1);
    auto ops = f->get_ordered_ops();
    auto res_count = std::count_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& n) {
        return std::dynamic_pointer_cast<ov::op::v0::Result>(n) != nullptr;
    });
    EXPECT_EQ(res_count, 1);
    auto names_count = std::count_if(ops.begin(), ops.end(), [](std::shared_ptr<ov::Node> n) {
        return n->output(0).get_tensor().get_names().count("tensor_output1") > 0;
    });
    EXPECT_EQ(names_count, 2);  // last node + result referencing to it
    EXPECT_EQ(name, f->output().get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
}

TEST(pre_post_process, postprocess_convert_element_type_default) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto name = f->output(1).get_node_shared_ptr()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names = f->output(1).get_tensor().get_names();
    f = PrePostProcessor(f)
            .output(OutputInfo(1)
                        .postprocess(PostProcessSteps().convert_element_type())
                        .tensor(OutputTensorInfo().set_element_type(element::u8)))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_results()[1]->get_element_type(), element::u8);
    EXPECT_EQ(name, f->output(1).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(tensor_names, f->output(1).get_tensor().get_names());
}

TEST(pre_post_process, postprocess_convert_element_type_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();
    f = PrePostProcessor(f)
            .output(OutputInfo("tensor_output1")
                        .postprocess(PostProcessSteps().convert_element_type(element::f32))
                        .tensor(OutputTensorInfo().set_element_type(element::f32)))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::f32);

    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, postprocess_convert_element_type_default_error) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    EXPECT_THROW(
        f = PrePostProcessor(f).output(OutputInfo().postprocess(PostProcessSteps().convert_element_type())).build(),
        ov::AssertFailure);
}

TEST(pre_post_process, postprocess_convert_element_type_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f).output(OutputInfo().tensor(OutputTensorInfo().set_element_type(element::u8))).build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
}

TEST(pre_post_process, preprocess_keep_params_order) {
    auto f = create_n_inputs<3>(element::f32, Shape{1, 2, 2, 3});
    f = PrePostProcessor(f)
            .input(InputInfo(1)
                       .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"}))
                       .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
            .input(InputInfo(0).tensor(InputTensorInfo().set_layout("NCHW")))
            .input(InputInfo(2)
                       .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"}))
                       .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
            .build();
    ASSERT_EQ(f->get_parameters().size(), 5);
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters()[1]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[2]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[3]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[4]->get_layout(), "NHWC");

    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{1, 1, 1, 2}));
    EXPECT_EQ(f->input(3).get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->input(4).get_partial_shape(), (PartialShape{1, 1, 1, 2}));

    EXPECT_EQ(f->input(0).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input0"});
    EXPECT_EQ(f->input(1).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input1/Y"});
    EXPECT_EQ(f->input(2).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input1/UV"});
    EXPECT_EQ(f->input(3).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input2/Y"});
    EXPECT_EQ(f->input(4).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input2/UV"});
}

// --- PostProcess - set/convert layout ---
TEST(pre_post_process, postprocess_set_layout_network) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f).output(OutputInfo().network(OutputNetworkInfo().set_layout("NCHW"))).build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW");
}

TEST(pre_post_process, postprocess_convert_layout_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    f = PrePostProcessor(f)
            .output(OutputInfo()
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .tensor(OutputTensorInfo().set_layout("NHWC")))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_explicit_no_target) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f)
            .output(OutputInfo(1)
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .postprocess(PostProcessSteps().convert_layout("NHWC")))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    EXPECT_EQ(f->get_results()[1]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    f = PrePostProcessor(f)
            .output(OutputInfo()
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .postprocess(PostProcessSteps().convert_layout())
                        .tensor(OutputTensorInfo().set_layout("NHWC")))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_default_getters) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    auto p = PrePostProcessor(f);
    auto& out = p.output();
    out.network().set_layout("NCHW");
    out.postprocess().convert_layout();
    out.tensor().set_layout("NHWC");
    f = p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();

    f = PrePostProcessor(f)
            .output(OutputInfo()
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .postprocess(PostProcessSteps().convert_layout("NCHW"))
                        .tensor(OutputTensorInfo().set_layout("NCHW")))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, postprocess_convert_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    f = PrePostProcessor(f).output(OutputInfo().postprocess(PostProcessSteps().convert_layout({0, 2, 3, 1}))).build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_dims_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    f = PrePostProcessor(f)
            .output(OutputInfo().postprocess(PostProcessSteps().convert_layout(std::vector<uint64_t>{})))
            .build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 3, 480, 640}));
}

TEST(pre_post_process, postprocess_convert_layout_has_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PostProcessSteps();
    p.convert_layout({0, 2, 3, 1});
    f = PrePostProcessor(f)
            .output(OutputInfo().network(OutputNetworkInfo().set_layout("NC??")).postprocess(std::move(p)))
            .build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_results()[0]->get_layout(), "N??C");
}

TEST(pre_post_process, postprocess_convert_layout_invalid_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    EXPECT_THROW(f = PrePostProcessor(f)
                         .output(OutputInfo().postprocess(PostProcessSteps().convert_layout({0, 3, 2, 2})))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .output(OutputInfo().postprocess(
                             PostProcessSteps().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()})))
                         .build(),
                 ov::AssertFailure);
}

TEST(pre_post_process, postprocess_convert_layout_invalid_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());
    EXPECT_THROW(f = PrePostProcessor(f)
                         .output(OutputInfo().postprocess(PostProcessSteps().convert_layout({0, 3, 2, 2})))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(f = PrePostProcessor(f)
                         .output(OutputInfo().postprocess(
                             PostProcessSteps().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()})))
                         .build(),
                 ov::AssertFailure);
}

// Postprocessing - other

TEST(pre_post_process, postprocess_custom_step) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    bool hit = false;
    f = PrePostProcessor(f)
            .output(OutputInfo().postprocess(PostProcessSteps().custom([&hit](const ov::Output<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                hit = true;
                return abs;
            })))
            .build();
    EXPECT_TRUE(hit);

    EXPECT_EQ(std::string(f->get_results()[0]->get_input_source_output(0).get_node()->get_type_name()),
              std::string(op::v0::Abs::get_type_info_static().name));
}

TEST(pre_post_process, postprocess_implicit_convert_element_type_and_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor(f)
            .output(OutputInfo()
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .tensor(OutputTensorInfo().set_layout("NHWC").set_element_type(element::u8)))
            .build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_assert_output_without_index) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto out = OutputInfo();
    EXPECT_ANY_THROW(f = PrePostProcessor(f).output(std::move(out)).build());
    out = OutputInfo("some_non_existing_name");
    EXPECT_ANY_THROW(f = PrePostProcessor(f).output(std::move(out)).build());
}

TEST(pre_post_process, postprocess_keep_results_order) {
    auto f = create_n_inputs<3>(element::f32, Shape{1, 3, 2, 2});
    auto names0 = f->output(0).get_tensor().get_names();
    auto names1 = f->output(1).get_tensor().get_names();
    auto names2 = f->output(2).get_tensor().get_names();
    f = PrePostProcessor(f)
            .output(OutputInfo(0).network(OutputNetworkInfo().set_layout("NCHW")))
            .output(OutputInfo(1)
                        .network(OutputNetworkInfo().set_layout("NCHW"))
                        .tensor(OutputTensorInfo().set_layout("NHWC").set_element_type(element::u8)))
            .build();
    ASSERT_EQ(f->get_results().size(), 3);
    EXPECT_EQ(f->output(0).get_element_type(), element::f32);
    EXPECT_EQ(f->output(1).get_element_type(), element::u8);
    EXPECT_EQ(f->output(2).get_element_type(), element::f32);

    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW") << f->get_results()[0]->get_layout().to_string();
    EXPECT_EQ(f->get_results()[1]->get_layout(), "NHWC") << f->get_results()[1]->get_layout().to_string();
    EXPECT_EQ(f->get_results()[2]->get_layout(), "") << f->get_results()[2]->get_layout().to_string();

    EXPECT_EQ(f->output(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    EXPECT_EQ(f->output(1).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(f->output(2).get_partial_shape(), (PartialShape{1, 3, 2, 2}));

    EXPECT_EQ(f->output(0).get_tensor().get_names(), names0);
    EXPECT_EQ(f->output(1).get_tensor().get_names(), names1);
    EXPECT_EQ(f->output(2).get_tensor().get_names(), names2);
}

TEST(pre_post_process, postprocess_lvalues_1) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    bool custom_called = false;

    auto netInfo = OutputNetworkInfo();
    netInfo.set_layout("NCHW");

    auto steps = PostProcessSteps();
    steps.convert_layout();
    steps.convert_element_type();
    steps.custom([&custom_called](const ov::Output<Node>& node) {
        custom_called = true;
        return std::make_shared<op::v0::Abs>(node);
    });

    auto tensorInfo = OutputTensorInfo();
    tensorInfo.set_layout("NHWC");
    tensorInfo.set_element_type(element::u8);

    auto outputInfo = OutputInfo("tensor_output1");
    outputInfo.network(std::move(netInfo));
    outputInfo.postprocess(std::move(steps));
    outputInfo.tensor(std::move(tensorInfo));

    auto p = PrePostProcessor(f);
    p.output(std::move(outputInfo));

    f = p.build();
    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->output().get_tensor().get_names().count("tensor_output1"), 1);
    EXPECT_EQ(f->output().get_node_shared_ptr()->get_friendly_name(), "Result1");
    EXPECT_EQ(f->output().get_element_type(), element::u8);
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_TRUE(custom_called);
}

TEST(pre_post_process, exception_safety) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 224, 224});
    auto name0 = f->input(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names0 = f->input(0).get_tensor().get_names();
    auto name1 = f->input(1).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names1 = f->input(1).get_tensor().get_names();
    auto out_name0 = f->output(0).get_node_shared_ptr()->get_friendly_name();
    auto out_tensor_names0 = f->output(0).get_tensor().get_names();
    auto out_name1 = f->output(1).get_node_shared_ptr()->get_friendly_name();
    auto out_tensor_names1 = f->output(1).get_tensor().get_names();
    EXPECT_THROW(f = PrePostProcessor(f)
                         .input(InputInfo(0)  // this one is correct
                                    .tensor(InputTensorInfo().set_element_type(element::u8))
                                    .preprocess(PreProcessSteps().convert_element_type(element::f32)))
                         .input(InputInfo(1)  // This one is not
                                    .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                                    .preprocess(PreProcessSteps().custom([](const Output<Node>& node) -> Output<Node> {
                                        throw ngraph::ngraph_error("test error");
                                    })))
                         .build(),
                 ov::AssertFailure);

    EXPECT_THROW(
        f = PrePostProcessor(f)
                .output(OutputInfo(0)  // this one is correct
                            .tensor(OutputTensorInfo().set_element_type(element::u8)))
                .output(OutputInfo(1)  // This one is not
                            .postprocess(PostProcessSteps().custom([](const Output<Node>& node) -> Output<Node> {
                                throw ngraph::ngraph_error("test error");
                            })))
                .build(),
        ngraph::ngraph_error);
    EXPECT_EQ(f->get_parameters().size(), 2);

    EXPECT_EQ(f->input(0).get_element_type(), element::f32);
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 3, 224, 224}));
    EXPECT_EQ(f->input(0).get_node_shared_ptr()->get_friendly_name(), name0);
    EXPECT_EQ(f->input(0).get_tensor().get_names(), tensor_names0);

    EXPECT_EQ(f->input(1).get_element_type(), element::f32);
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 3, 224, 224}));
    EXPECT_EQ(f->input(1).get_node_shared_ptr()->get_friendly_name(), name1);
    EXPECT_EQ(f->input(1).get_tensor().get_names(), tensor_names1);

    EXPECT_EQ(f->output(0).get_node_shared_ptr()->get_friendly_name(), out_name0);
    EXPECT_EQ(f->output(0).get_tensor().get_names(), out_tensor_names0);

    EXPECT_EQ(f->output(1).get_node_shared_ptr()->get_friendly_name(), out_name1);
    EXPECT_EQ(f->output(1).get_tensor().get_names(), out_tensor_names1);
}
