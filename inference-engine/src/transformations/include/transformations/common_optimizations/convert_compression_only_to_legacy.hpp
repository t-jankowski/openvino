// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations_visibility.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertPrecisionCompressedOnly;
class TRANSFORMATIONS_API EnableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API ConvertCompressedOnlyToLegacy;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertPrecisionCompressedOnly transformation runs ConvertPrecision transformation for CompressedOnly format.
 */

class ov::pass::ConvertPrecisionCompressedOnly : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertPrecisionCompressedOnly", "0");
    bool run_on_function(std::shared_ptr<Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Enables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::EnableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EnableDecompressionConvertConstantFolding", "0");
    EnableDecompressionConvertConstantFolding();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertCompressedOnlyToLegacy transformation converts compression only FP16 format to legacy FP16 format.
 */
class ov::pass::ConvertCompressedOnlyToLegacy : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertCompressedOnlyToLegacy", "0");
    bool run_on_function(std::shared_ptr<Function> f) override;
};
