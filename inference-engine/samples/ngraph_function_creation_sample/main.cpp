// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "format_reader_ptr.h"
#include "gflags/gflags.h"
#include "ngraph/util.hpp"
#include "ngraph_function_creation_sample.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset8.hpp"
#include "samples/args_helper.hpp"
#include "samples/classification_results.h"
#include "samples/common.hpp"
#include "samples/slog.hpp"

using namespace ov;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_nt <= 0 || FLAGS_nt > 10) {
        throw std::logic_error("Incorrect value for nt argument. It should be "
                               "greater than 0 and less than 10.");
    }

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Path to a .bin file with weights for the trained model is required "
                               "but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        showUsage();
        throw std::logic_error("Path to an image is required but not set. Please set -i option.");
    }

    return true;
}

/**
 * @brief Read file to the buffer
 * @param file_name string
 * @param buffer to store file content
 * @param maxSize length of file
 * @return none
 */
void readFile(const std::string& file_name, void* buffer, size_t maxSize) {
    std::ifstream input_file;

    input_file.open(file_name, std::ios::binary | std::ios::in);
    if (!input_file.is_open()) {
        throw std::logic_error("Cannot open weights file");
    }

    if (!input_file.read(reinterpret_cast<char*>(buffer), maxSize)) {
        input_file.close();
        throw std::logic_error("Cannot read bytes from weights file");
    }

    input_file.close();
}

/**
 * @brief Read .bin file with weights for the trained model
 * @param filepath string
 * @return weightsPtr tensor blob
 */
ov::runtime::Tensor ReadWeights(const std::string& filepath) {
    std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);

    int64_t fileSize = weightFile.tellg();
    OPENVINO_ASSERT(fileSize == 1724336,
                    "Incorrect weights file. This sample works only with LeNet "
                    "classification model.");

    ov::runtime::Tensor weights(ov::element::u8, {static_cast<size_t>(fileSize)});
    readFile(filepath, weights.data(), weights.get_byte_size());

    return std::move(weights);
}

/**
 * @brief Create ngraph function
 * @return Ptr to ngraph function
 */
std::shared_ptr<ov::Function> createNgraphFunction() {
    auto weights = ReadWeights(FLAGS_m);
    const std::uint8_t* data = weights.data<std::uint8_t>();

    // -------input------
    std::vector<ptrdiff_t> padBegin{0, 0};
    std::vector<ptrdiff_t> padEnd{0, 0};

    auto paramNode = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape({64, 1, 28, 28}));

    // -------convolution 1----
    auto convFirstShape = Shape{20, 1, 5, 5};
    auto convolutionFirstConstantNode = std::make_shared<opset8::Constant>(element::Type_t::f32, convFirstShape, data);

    auto convolutionNodeFirst = std::make_shared<opset8::Convolution>(paramNode->output(0),
                                                                      convolutionFirstConstantNode->output(0),
                                                                      Strides({1, 1}),
                                                                      CoordinateDiff(padBegin),
                                                                      CoordinateDiff(padEnd),
                                                                      Strides({1, 1}));

    // -------Add--------------
    auto addFirstShape = Shape{1, 20, 1, 1};
    auto offset = shape_size(convFirstShape) * sizeof(float);
    auto addFirstConstantNode = std::make_shared<opset8::Constant>(element::Type_t::f32, addFirstShape, data + offset);

    auto addNodeFirst = std::make_shared<opset8::Add>(convolutionNodeFirst->output(0), addFirstConstantNode->output(0));

    // -------MAXPOOL----------
    Shape padBeginShape{0, 0};
    Shape padEndShape{0, 0};

    auto maxPoolingNodeFirst = std::make_shared<op::v1::MaxPool>(addNodeFirst->output(0),
                                                                 Strides{2, 2},
                                                                 padBeginShape,
                                                                 padEndShape,
                                                                 Shape{2, 2},
                                                                 op::RoundingType::CEIL);

    // -------convolution 2----
    auto convSecondShape = Shape{50, 20, 5, 5};
    offset += shape_size(addFirstShape) * sizeof(float);
    auto convolutionSecondConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::f32, convSecondShape, data + offset);

    auto convolutionNodeSecond = std::make_shared<opset8::Convolution>(maxPoolingNodeFirst->output(0),
                                                                       convolutionSecondConstantNode->output(0),
                                                                       Strides({1, 1}),
                                                                       CoordinateDiff(padBegin),
                                                                       CoordinateDiff(padEnd),
                                                                       Strides({1, 1}));

    // -------Add 2------------
    auto addSecondShape = Shape{1, 50, 1, 1};
    offset += shape_size(convSecondShape) * sizeof(float);
    auto addSecondConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::f32, addSecondShape, data + offset);

    auto addNodeSecond =
        std::make_shared<opset8::Add>(convolutionNodeSecond->output(0), addSecondConstantNode->output(0));

    // -------MAXPOOL 2--------
    auto maxPoolingNodeSecond = std::make_shared<op::v1::MaxPool>(addNodeSecond->output(0),
                                                                  Strides{2, 2},
                                                                  padBeginShape,
                                                                  padEndShape,
                                                                  Shape{2, 2},
                                                                  op::RoundingType::CEIL);

    // -------Reshape----------
    auto reshapeFirstShape = Shape{2};
    auto reshapeOffset = shape_size(addSecondShape) * sizeof(float) + offset;
    auto reshapeFirstConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::i64, reshapeFirstShape, data + reshapeOffset);

    auto reshapeFirstNode =
        std::make_shared<op::v1::Reshape>(maxPoolingNodeSecond->output(0), reshapeFirstConstantNode->output(0), true);

    // -------MatMul 1---------
    auto matMulFirstShape = Shape{500, 800};
    offset = shape_size(reshapeFirstShape) * sizeof(int64_t) + reshapeOffset;
    auto matMulFirstConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::f32, matMulFirstShape, data + offset);

    auto matMulFirstNode =
        std::make_shared<opset8::MatMul>(reshapeFirstNode->output(0), matMulFirstConstantNode->output(0), false, true);

    // -------Add 3------------
    auto addThirdShape = Shape{1, 500};
    offset += shape_size(matMulFirstShape) * sizeof(float);
    auto addThirdConstantNode = std::make_shared<opset8::Constant>(element::Type_t::f32, addThirdShape, data + offset);

    auto addThirdNode = std::make_shared<opset8::Add>(matMulFirstNode->output(0), addThirdConstantNode->output(0));

    // -------Relu-------------
    auto reluNode = std::make_shared<opset8::Relu>(addThirdNode->output(0));

    // -------Reshape 2--------
    auto reshapeSecondShape = Shape{2};
    auto reshapeSecondConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::i64, reshapeSecondShape, data + reshapeOffset);

    auto reshapeSecondNode =
        std::make_shared<op::v1::Reshape>(reluNode->output(0), reshapeSecondConstantNode->output(0), true);

    // -------MatMul 2---------
    auto matMulSecondShape = Shape{10, 500};
    offset += shape_size(addThirdShape) * sizeof(float);
    auto matMulSecondConstantNode =
        std::make_shared<opset8::Constant>(element::Type_t::f32, matMulSecondShape, data + offset);

    auto matMulSecondNode = std::make_shared<opset8::MatMul>(reshapeSecondNode->output(0),
                                                             matMulSecondConstantNode->output(0),
                                                             false,
                                                             true);

    // -------Add 4------------
    auto add4Shape = Shape{1, 10};
    offset += shape_size(matMulSecondShape) * sizeof(float);
    auto add4ConstantNode = std::make_shared<opset8::Constant>(element::Type_t::f32, add4Shape, data + offset);

    auto add4Node = std::make_shared<opset8::Add>(matMulSecondNode->output(0), add4ConstantNode->output(0));

    // -------softMax----------
    auto softMaxNode = std::make_shared<opset8::Softmax>(add4Node->output(0), 1);
    softMaxNode->get_output_tensor(0).set_names({"output_tensor"});

    // ------- OpenVINO function--
    auto result_full = std::make_shared<opset8::Result>(softMaxNode->output(0));

    std::shared_ptr<ov::Function> fnPtr =
        std::make_shared<ov::Function>(result_full, ov::ParameterVector{paramNode}, "lenet");

    return fnPtr;
}

/**
 * @brief The entry point for inference engine automatic ov::Function
 * creation sample
 * @file ngraph_function_creation_sample/main.cpp
 * @example ngraph_function_creation_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << "OpenVINO Runtime: " << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        // -------- Read input --------
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        OPENVINO_ASSERT(!images.empty(), "No suitable images were found");

        // -------- Step 1. Initialize OpenVINO Runtime Core object --------
        slog::info << "Loading OpenVINO runtime" << slog::endl;
        runtime::Core core;

        slog::info << "Device info: " << slog::endl;
        std::cout << core.get_versions(FLAGS_d) << std::endl;

        // -------- Step 2. Create network using ov::Function --------

        auto model = createNgraphFunction();

        // -------- Step 3. Apply preprocessing --------
        const Layout tensor_layout{"NHWC"};

        // apply preprocessing
        // clang-format off
        using namespace ov::preprocess;
        model = PrePostProcessor(model)
            // 1) InputInfo() with no args assumes a model has a single input
            .input(InputInfo()
                // 2) Set input tensor information:
                // - precision of tensor is supposed to be 'u8'
                // - layout of data is 'NHWC'
                .tensor(InputTensorInfo()
                    .set_layout(tensor_layout)
                    .set_element_type(element::u8))
                // 3) Here we suppose model has 'NCHW' layout for input
                .network(InputNetworkInfo()
                    .set_layout("NCHW")))
        // 4) Once the build() method is called, the preprocessing steps
        // for layout and precision conversions are inserted automatically
        .build();
        // clang-format on

        // -------- Step 4. Read input images --------

        const auto input = model->input();

        auto input_shape = input.get_shape();
        const size_t width = input_shape[layout::width_idx(tensor_layout)];
        const size_t height = input_shape[layout::height_idx(tensor_layout)];

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto& i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }

            if (reader->size() != width * height) {
                throw std::logic_error("Not supported format. Only MNist ubyte images supported.");
            }

            // Store image data
            std::shared_ptr<unsigned char> data(reader->getData(width, height));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }

        OPENVINO_ASSERT(!imagesData.empty(), "Valid input images were not found");

        // -------- Step 4. Reshape a model --------
        // Setting batch size using image count
        const size_t batch_size = imagesData.size();
        input_shape[layout::batch_idx(tensor_layout)] = batch_size;
        model->reshape({{input.get_any_name(), input_shape}});
        slog::info << "Batch size is " << std::to_string(batch_size) << slog::endl;

        const auto outputShape = model->output().get_shape();
        OPENVINO_ASSERT(outputShape.size() == 2, "Incorrect output dimensions for LeNet");

        const auto classCount = outputShape[1];
        OPENVINO_ASSERT(classCount <= 10, "Incorrect number of output classes for LeNet model");

        // -------- Step 4. Compiling model for the device --------
        slog::info << "Compiling a model for the " << FLAGS_d << " device" << slog::endl;
        runtime::ExecutableNetwork exeNetwork = core.compile_model(model, FLAGS_d);

        // -------- Step 5. Create infer request --------
        slog::info << "Create infer request" << slog::endl;
        runtime::InferRequest infer_request = exeNetwork.create_infer_request();

        // -------- Step 6. Combine multiple input images as batch --------
        slog::info << "Combining a batch and set input tensor" << slog::endl;
        runtime::Tensor input_tensor = infer_request.get_input_tensor();

        // Iterate over all input images
        for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
            const size_t image_size = shape_size(input_shape) / batch_size;
            std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size,
                        imagesData[image_id].get(),
                        image_size);
        }

        // -------- Step 7. Do sync inference --------
        slog::info << "Start sync inference" << slog::endl;
        infer_request.infer();

        // -------- Step 8. Process output --------
        slog::info << "Processing output tensor" << slog::endl;
        const runtime::Tensor output_tensor = infer_request.get_output_tensor();

        // Validating -nt value
        const size_t results_cnt = output_tensor.get_size() / batch_size;
        if (FLAGS_nt > results_cnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this model (-nt should be less than "
                       << results_cnt + 1 << " and more than 0).\n           Maximal value " << results_cnt
                       << " will be used.";
            FLAGS_nt = results_cnt;
        }

        // Read labels from file (e.x. LeNet.labels) **/
        std::string label_file_name = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;

        std::ifstream input_file;
        input_file.open(label_file_name, std::ios::in);
        if (input_file.is_open()) {
            std::string strLine;
            while (std::getline(input_file, strLine)) {
                trim(strLine);
                labels.push_back(strLine);
            }
            input_file.close();
        }

        // Prints formatted classification results
        ClassificationResult classification_result(output_tensor, images, batch_size, FLAGS_nt, labels);
        classification_result.show();
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }

    slog::info << "This sample is an API example, for performance measurements, "
                  "use the dedicated benchmark_app tool"
               << slog::endl;

    return EXIT_SUCCESS;
}
