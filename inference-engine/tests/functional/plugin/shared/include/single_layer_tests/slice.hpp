// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/slice.hpp"

namespace LayerTestsDefinitions {
TEST_P(SliceLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
