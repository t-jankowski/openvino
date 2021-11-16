// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"

#include "transformations/convert_precision.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "itt.hpp"

using namespace ov;

bool ov::pass::ConvertPrecisionCompressedOnly::run_on_function(std::shared_ptr<ov::Function> f) {
    if (ngraph::op::util::has_decompression_converts(f)) {
        const precisions_array convert_precision_list{
            {ov::element::f32, ov::element::f16}
        };
        auto convert_precision = ngraph::pass::ConvertPrecision(convert_precision_list);
        return convert_precision.run_on_function(f);
    }
    return false;
}

ov::pass::EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<opset8::Convert>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!ov::is_decompression(node))
            return false;
        enable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::pass::ConvertCompressedOnlyToLegacy::run_on_function(std::shared_ptr<ov::Function> f) {
    Manager manager(get_pass_config());

    manager.register_pass<ov::pass::ConvertPrecisionCompressedOnly>();
    manager.register_pass<ov::pass::EnableDecompressionConvertConstantFolding>();
    manager.register_pass<ov::pass::ConstantFolding>();

    manager.run_passes(f);

    return false;
}
