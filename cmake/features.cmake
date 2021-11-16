# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common cmake options
#

ie_dependent_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON "X86_64" OFF)

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_STRICT_DEPENDENCIES "Skip configuring \"convinient\" dependencies for efficient parallel builds" ON)

ie_dependent_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON "X86_64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

if (NOT ENABLE_CLDNN OR ANDROID OR
    (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and android builds for now, so we'll
    # build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ie_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ON "${ENABLE_ONEDNN_FOR_GPU_DEFAULT}" OFF)

ie_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ie_option_enum(ENABLE_PROFILING_FILTER "Enable or disable ITT counter groups.\
Supported values:\
 ALL - enable all ITT counters (default value)\
 FIRST_INFERENCE - enable only first inference time counters" ALL
               ALLOWED_VALUES ALL FIRST_INFERENCE)

ie_option (ENABLE_PROFILING_FIRST_INFERENCE "Build with ITT tracing of first inference time." ON)

ie_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)

ie_option(ENABLE_ERROR_HIGHLIGHT "Highlight errors and warnings during compile time" OFF)

# Try to find python3
find_package(PythonLibs 3 QUIET)
ie_dependent_option (ENABLE_PYTHON "enables ie python bridge build" OFF "PYTHONLIBS_FOUND" OFF)

find_package(PythonInterp 3 QUIET)
ie_dependent_option (ENABLE_DOCS "Build docs using Doxygen" OFF "PYTHONINTERP_FOUND" OFF)

#
# Inference Engine specific options
#

ie_dependent_option (ENABLE_GNA "GNA support for inference engine" ON "NOT APPLE;NOT ANDROID;X86_64" OFF)

ie_dependent_option (ENABLE_CLDNN_TESTS "Enable clDNN unit tests" OFF "ENABLE_CLDNN" OFF)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if(X86 OR ARM OR (MSVC AND (ARM OR AARCH64)) )
    set(THREADING_DEFAULT "SEQ")
else()
    set(THREADING_DEFAULT "TBB")
endif()
set(THREADING "${THREADING_DEFAULT}" CACHE STRING "Threading")
set_property(CACHE THREADING PROPERTY STRINGS "TBB" "TBB_AUTO" "OMP" "SEQ")
list (APPEND IE_OPTIONS THREADING)
if (NOT THREADING STREQUAL "TBB" AND
    NOT THREADING STREQUAL "TBB_AUTO" AND
    NOT THREADING STREQUAL "OMP" AND
    NOT THREADING STREQUAL "SEQ")
    message(FATAL_ERROR "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is ${THREADING_DEFAULT}")
endif()

if (ENABLE_GNA)
    if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        set (DEFAULT_GNA_LIB GNA1)
    else()
        set (DEFAULT_GNA_LIB GNA2)
    endif()
    set(GNA_LIBRARY_VERSION "${DEFAULT_GNA_LIB}" CACHE STRING "GNAVersion")
    set_property(CACHE GNA_LIBRARY_VERSION PROPERTY STRINGS "GNA1" "GNA1_1401" "GNA2")
    list (APPEND IE_OPTIONS GNA_LIBRARY_VERSION)
    if (NOT GNA_LIBRARY_VERSION STREQUAL "GNA1" AND
        NOT GNA_LIBRARY_VERSION STREQUAL "GNA1_1401" AND
        NOT GNA_LIBRARY_VERSION STREQUAL "GNA2")
        message(FATAL_ERROR "GNA_LIBRARY_VERSION should be set to GNA1, GNA1_1401 or GNA2. Default option is ${DEFAULT_GNA_LIB}")
    endif()
endif()

if(ENABLE_TESTS OR BUILD_SHARED_LIBS)
    set(ENABLE_IR_V7_READER_DEFAULT ON)
else()
    set(ENABLE_IR_V7_READER_DEFAULT OFF)
endif()

ie_option (ENABLE_IR_V7_READER "Enables IR v7 reader" ${ENABLE_IR_V7_READER_DEFAULT})

ie_option (ENABLE_MULTI "Enables Multi Device Plugin" ON)

ie_option (ENABLE_HETERO "Enables Hetero Device Plugin" ON)

ie_option (ENABLE_TEMPLATE "Enable template plugin" ON)

ie_dependent_option (ENABLE_VPU "vpu targeted plugins for inference engine" ON "NOT WINDOWS_PHONE;NOT WINDOWS_STORE" OFF)

ie_dependent_option (ENABLE_MYRIAD "myriad targeted plugin for inference engine" ON "ENABLE_VPU" OFF)

ie_dependent_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF "ENABLE_MYRIAD" OFF)

ie_dependent_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" ON "ENABLE_TESTS" OFF)

ie_dependent_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF "ENABLE_GAPI_TESTS" OFF)

ie_dependent_option (ENABLE_MYRIAD_MVNC_TESTS "functional and behavior tests for mvnc api" OFF "ENABLE_TESTS;ENABLE_MYRIAD" OFF)

ie_dependent_option (ENABLE_DATA "fetch models from testdata repo" ON "ENABLE_FUNCTIONAL_TESTS;NOT ANDROID" OFF)

ie_dependent_option (ENABLE_BEH_TESTS "tests oriented to check inference engine API corecteness" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON "NOT MINGW" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (ENABLE_V7_SERIALIZE "enables serialization to IR v7" OFF)

set(IE_EXTRA_MODULES "" CACHE STRING "Extra paths for extra modules to include into OpenVINO build")

ie_dependent_option(ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the Inference Engine binaries" ON "THREADING MATCHES TBB;LINUX" OFF)

ie_dependent_option (ENABLE_SYSTEM_PUGIXML "use the system copy of pugixml" OFF "BUILD_SHARED_LIBS" OFF)

ie_option (ENABLE_DEBUG_CAPS "enable OpenVINO debug capabilities at runtime" OFF)

ie_dependent_option (ENABLE_GPU_DEBUG_CAPS "enable GPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

ie_dependent_option (ENABLE_CPU_DEBUG_CAPS "enable CPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

if(ANDROID OR WINDOWS_STORE OR (MSVC AND (ARM OR AARCH64)))
    set(protoc_available OFF)
else()
    set(protoc_available ON)
endif()

ie_dependent_option(NGRAPH_ONNX_FRONTEND_ENABLE "Enable ONNX FrontEnd" ON "protoc_available" OFF)
ie_dependent_option(NGRAPH_PDPD_FRONTEND_ENABLE "Enable PaddlePaddle FrontEnd" ON "protoc_available" OFF)
ie_option(NGRAPH_IR_FRONTEND_ENABLE "Enable IR FrontEnd" ON)
ie_dependent_option(NGRAPH_TF_FRONTEND_ENABLE "Enable TensorFlow FrontEnd" ON "protoc_available" OFF)
ie_dependent_option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system protobuf" OFF
    "NGRAPH_ONNX_FRONTEND_ENABLE OR NGRAPH_PDPD_FRONTEND_ENABLE OR NGRAPH_TF_FRONTEND_ENABLE" OFF)
ie_dependent_option(NGRAPH_UNIT_TEST_ENABLE "Enables ngraph unit tests" ON "ENABLE_TESTS;NOT ANDROID" OFF)
ie_dependent_option(NGRAPH_UNIT_TEST_BACKENDS_ENABLE "Control the building of unit tests using backends" ON
    "NGRAPH_UNIT_TEST_ENABLE" OFF)
ie_option(OPENVINO_DEBUG_ENABLE "Enable output for OPENVINO_DEBUG statements" OFF)
ie_option(ENABLE_REQUIREMENTS_INSTALL "Dynamic dependencies install" ON)

# WA for ngraph python build on Windows debug
list(REMOVE_ITEM IE_OPTIONS NGRAPH_UNIT_TEST_ENABLE NGRAPH_UNIT_TEST_BACKENDS_ENABLE)

#
# Process featues
#

if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_MYRIAD)
    add_definitions(-DENABLE_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_MYRIAD )
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_MKL_DNN)
    add_definitions(-DENABLE_MKL_DNN=1)
endif()

if (ENABLE_GNA)
    add_definitions(-DENABLE_GNA)

    if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(WARNING "${GNA_LIBRARY_VERSION} is not supported on GCC version ${CMAKE_CXX_COMPILER_VERSION}. Fallback to GNA1")
        set(GNA_LIBRARY_VERSION GNA1)
    endif()
endif()

print_enabled_features()
