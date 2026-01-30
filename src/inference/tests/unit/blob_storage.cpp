// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef BLOBS_STORAGE_ENABLED
#    include "../../src/dev/blob_storage.hpp"

#    include <gtest/gtest.h>

#    include "openvino/util/variant_visitor.hpp"

namespace ov::test {

TEST(BlobStorageTest, BundleReadFromVector) {
    using namespace ov::storage;

    const std::string test_text{"read from vector test"};
    const std::vector<Bundle::byte_type> test_value(test_text.begin(), test_text.end());
    std::stringstream storage;
    {
        auto moveable_test_value = test_value;
        std::stringstream test_stream;
        test_stream << test_text;

        BundlePool bundles_to_keep;
        bundles_to_keep.add_entry(0, test_value);
        bundles_to_keep.add_entry(1, std::move(moveable_test_value));
        bundles_to_keep.add_entry(2, &test_stream);
        bundles_to_keep.add_entry(3, test_text.size(), test_text.data());

        bundles_to_keep.write_to(storage);
    }

    // Read back and verify
    BundlePool restored_bundles;
    storage.seekg(0, std::ios::beg);
    restored_bundles.read_from(storage);

    const auto& bundles = restored_bundles.pool();
    ASSERT_EQ(bundles.size(), 4);

    for (size_t i = 0; i < bundles.size(); ++i) {
        const auto& bundle = bundles[i];
        EXPECT_EQ(bundle.tag, i);
        EXPECT_EQ(bundle.length, test_value.size());

        const auto test_read =
            ov::util::VariantVisitor{[](const Bundle::BufferView&) {
                                         FAIL() << "Expected std::vector, got BufferView";
                                     },
                                     [](std::istream*) {
                                         FAIL() << "Expected std::vector, got istream*";
                                     },
                                     [&test_value](const std::vector<Bundle::byte_type>& read_value) {
                                         EXPECT_EQ(read_value, test_value);
                                     }};
        std::visit(test_read, bundle.value);
    }
}

TEST(BlobStorageTest, BundleReadFromStream) {
    using namespace ov::storage;
    GTEST_SKIP() << "feature is under development";

    const std::string test_text{"read from stream test"};
    std::stringstream storage;
    {
        std::stringstream test_stream;
        test_stream << test_text;

        BundlePool bundles_to_keep;
        bundles_to_keep.add_entry(100, &test_stream);

        bundles_to_keep.write_to(storage);
    }

    // Read back and verify
    BundlePool restored_bundles;
    storage.seekg(0, std::ios::beg);
    restored_bundles.read_from(storage);

    const auto& bundles = restored_bundles.pool();
    ASSERT_EQ(bundles.size(), 1);
    const auto& bundle = bundles[0];
    EXPECT_EQ(bundle.tag, 100);
    EXPECT_EQ(bundle.length, test_text.size());

    const auto read_visitor = ov::util::VariantVisitor{[](const Bundle::BufferView&) {
                                                           FAIL() << "Expected istream*, got BufferView";
                                                       },
                                                       [&test_text](std::istream* read_value) {
                                                           std::string read_text;
                                                           (*read_value) >> read_text;
                                                           EXPECT_EQ(read_text, test_text);
                                                       },
                                                       [](const std::vector<Bundle::byte_type>&) {
                                                           FAIL() << "Expected istream*, got vector";
                                                       }};
    std::visit(read_visitor, bundle.value);
}

}  // namespace ov::test
#endif  // BLOBS_STORAGE_ENABLED
