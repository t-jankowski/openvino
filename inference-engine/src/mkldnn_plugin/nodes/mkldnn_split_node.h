// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNSplitNode : public MKLDNNNode {
public:
    MKLDNNSplitNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool isOptimized() const;
    void initOptimalPrimitiveDescriptor() override;

    void setDynamicBatchLim(int lim) override;
    bool isExecutable() const override {
        return !isOptimized();
    }

    bool needPrepareParams() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }

private:
    struct SplitExecutor {
        virtual void exec(const uint8_t* srcData, const std::vector<uint8_t*> &dstMemPtrs,
                          const Dim origBatch, const Dim perInferBatch) = 0;
        virtual ~SplitExecutor() = default;
    };
    std::shared_ptr<SplitExecutor> execPtr = nullptr;

    struct SplitOptimizedExecutor : public SplitExecutor {
        public:
            SplitOptimizedExecutor(BlockedMemoryDescCPtr inDesc, const std::vector<BlockedMemoryDescCPtr> &outDescs, const size_t axis);
            void exec(const uint8_t* srcData, const std::vector<uint8_t*> &dstMemPtrs,
                      const Dim origBatch, const Dim perInferBatch) override;

        private:
            std::vector<size_t> dataSize;
            std::vector<size_t> srcDataOffsets;
            size_t srcDataStride;
            size_t countStrides;
    };

    void optimizedNspc2Ncsp(size_t MB);

    bool canUseOptimizedNspc2Ncsp;

    size_t axis = 1;
    std::vector<uint8_t*> dstMemPtrs;

    size_t INPUTS_NUM = 2;
};

}  // namespace MKLDNNPlugin

