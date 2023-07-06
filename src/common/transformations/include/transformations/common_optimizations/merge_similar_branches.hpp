// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @brief
 * @ingroup ov_pass_cpp_api
 */
class TRANSFORMATIONS_API MergeSimilarBranches : public ModelPass {
public:
    OPENVINO_RTTI("MergeSimilarBranches");
    MergeSimilarBranches() = default;
    MergeSimilarBranches(bool identical, bool matmul_add, bool matmul)
        : m_identical{identical},
          m_matmul_add{matmul_add},
          m_matmul{matmul} {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_identical{true};
    bool m_matmul_add{true};
    bool m_matmul{true};
};

}  // namespace pass
}  // namespace ov
