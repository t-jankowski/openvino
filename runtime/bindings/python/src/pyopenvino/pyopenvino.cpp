// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <openvino/core/node.hpp>
#include <openvino/core/version.hpp>
#include <string>

#include "pyopenvino/graph/axis_set.hpp"
#include "pyopenvino/graph/axis_vector.hpp"
#include "pyopenvino/graph/coordinate.hpp"
#include "pyopenvino/graph/coordinate_diff.hpp"
#include "pyopenvino/graph/function.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/node_factory.hpp"
#include "pyopenvino/graph/node_input.hpp"
#include "pyopenvino/graph/node_output.hpp"
#if defined(NGRAPH_ONNX_FRONTEND_ENABLE)
#    include "pyopenvino/graph/onnx_import/onnx_import.hpp"
#endif
#include "pyopenvino/core/async_infer_queue.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/core.hpp"
#include "pyopenvino/core/executable_network.hpp"
#include "pyopenvino/core/ie_blob.hpp"
#include "pyopenvino/core/ie_data.hpp"
#include "pyopenvino/core/ie_input_info.hpp"
#include "pyopenvino/core/ie_network.hpp"
#include "pyopenvino/core/ie_parameter.hpp"
#include "pyopenvino/core/ie_preprocess_info.hpp"
#include "pyopenvino/core/infer_request.hpp"
#include "pyopenvino/core/offline_transformations.hpp"
#include "pyopenvino/core/profiling_info.hpp"
#include "pyopenvino/core/tensor.hpp"
#include "pyopenvino/core/tensor_description.hpp"
#include "pyopenvino/core/version.hpp"
#include "pyopenvino/graph/dimension.hpp"
#include "pyopenvino/graph/layout.hpp"
#include "pyopenvino/graph/ops/constant.hpp"
#include "pyopenvino/graph/ops/parameter.hpp"
#include "pyopenvino/graph/ops/result.hpp"
#include "pyopenvino/graph/ops/util/regmodule_graph_op_util.hpp"
#include "pyopenvino/graph/partial_shape.hpp"
#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/graph/shape.hpp"
#include "pyopenvino/graph/strides.hpp"
#include "pyopenvino/graph/types/regmodule_graph_types.hpp"
#include "pyopenvino/graph/util.hpp"
#include "pyopenvino/graph/variant.hpp"

namespace py = pybind11;

std::string get_version() {
    auto version = ov::get_openvino_version();
    std::string version_str = std::to_string(OPENVINO_VERSION_MAJOR) + ".";
    version_str += std::to_string(OPENVINO_VERSION_MINOR) + ".";
    version_str += version->buildNumber;
    return version_str;
}

PYBIND11_MODULE(pyopenvino, m) {
    m.doc() = "Package openvino.pyopenvino which wraps openvino C++ APIs";
    m.def("get_version", &get_version);

    regclass_graph_PyRTMap(m);
    regmodule_graph_types(m);
    regclass_graph_Dimension(m);  // Dimension must be registered before PartialShape
    regclass_graph_Layout(m);
    regclass_graph_Shape(m);
    regclass_graph_PartialShape(m);
    regclass_graph_Node(m);
    regclass_graph_Input(m);
    regclass_graph_NodeFactory(m);
    regclass_graph_Strides(m);
    regclass_graph_CoordinateDiff(m);
    regclass_graph_AxisSet(m);
    regclass_graph_AxisVector(m);
    regclass_graph_Coordinate(m);
    py::module m_op = m.def_submodule("op", "Package ngraph.impl.op that wraps ov::op");  // TODO(!)
    regclass_graph_op_Constant(m_op);
    regclass_graph_op_Parameter(m_op);
    regclass_graph_op_Result(m_op);
#if defined(NGRAPH_ONNX_FRONTEND_ENABLE)
    regmodule_graph_onnx_import(m);
#endif
    regmodule_graph_op_util(m_op);
    regclass_graph_Function(m);
    regmodule_graph_passes(m);
    regmodule_graph_util(m);
    regclass_graph_Variant(m);
    regclass_graph_VariantWrapper<std::string>(m, std::string("String"));
    regclass_graph_VariantWrapper<int64_t>(m, std::string("Int"));
    regclass_graph_Output<ov::Node>(m, std::string(""));
    regclass_graph_Output<const ov::Node>(m, std::string("Const"));

    regclass_Core(m);
    regclass_IENetwork(m);

    regclass_Data(m);
    regclass_TensorDecription(m);

    // Blob will be removed
    // Registering template of Blob
    regclass_Blob(m);
    // Registering specific types of Blobs
    regclass_TBlob<float>(m, "Float32");
    regclass_TBlob<double>(m, "Float64");
    regclass_TBlob<int64_t>(m, "Int64");
    regclass_TBlob<uint64_t>(m, "Uint64");
    regclass_TBlob<int32_t>(m, "Int32");
    regclass_TBlob<uint32_t>(m, "Uint32");
    regclass_TBlob<int16_t>(m, "Int16");
    regclass_TBlob<uint16_t>(m, "Uint16");
    regclass_TBlob<int8_t>(m, "Int8");
    regclass_TBlob<uint8_t>(m, "Uint8");

    regclass_Tensor(m);

    // Registering specific types of containers
    Containers::regclass_TensorIndexMap(m);
    Containers::regclass_TensorNameMap(m);

    regclass_ExecutableNetwork(m);
    regclass_InferRequest(m);
    regclass_Version(m);
    regclass_Parameter(m);
    regclass_InputInfo(m);
    regclass_AsyncInferQueue(m);
    regclass_ProfilingInfo(m);
    regclass_PreProcessInfo(m);

    regmodule_offline_transformations(m);
}
