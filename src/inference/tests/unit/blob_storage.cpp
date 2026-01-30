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

    const std::string test_text{"Blob Storage test"};
    const std::vector<Bundle::byte_t> test_value(test_text.begin(), test_text.end());
    auto moveable_test_value = test_value;
    std::stringstream test_stream;
    test_stream << test_text;

    BundlePool pool;
    pool.add_entry(100, test_value);
    pool.add_entry(200, std::move(moveable_test_value));
    pool.add_entry(300, &test_stream);
    pool.add_entry(400, test_text.size(), test_text.data());

    std::stringstream ss;
    pool.write_to(ss);

    // Read back and verify
    BundlePool read_pool;
    ss.seekg(0, std::ios::beg);
    read_pool.read_from(ss);

    const auto& entries = read_pool.entries();
    ASSERT_EQ(entries.size(), 4);

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        EXPECT_EQ(entry.tag, 100 + i * 100);
        EXPECT_EQ(entry.length, test_value.size());

        const auto read_visitor =
            ov::util::VariantVisitor{[](const Bundle::BufferView&) {
                                         FAIL() << "Expected std::vector, got BufferView";
                                     },
                                     [](std::stringstream* ss) {
                                         FAIL() << "Expected std::vector, got stringstream*";
                                     },
                                     [&test_value](const std::vector<Bundle::byte_t>& read_value) {
                                         EXPECT_EQ(read_value, test_value);
                                     }};
        std::visit(read_visitor, entry.value);
    }
}

TEST(BlobStorageTest, BundleReadFromStream) {
    using namespace ov::storage;
    GTEST_SKIP() << "feature is under development";

    const std::string test_text{"Blob Storage stream test"};
    std::stringstream test_stream;
    test_stream << test_text;

    BundlePool pool;
    pool.add_entry(100, &test_stream);

    std::stringstream ss;
    pool.write_to(ss);

    // Read back and verify
    BundlePool read_pool;
    ss.seekg(0, std::ios::beg);
    read_pool.read_from(ss);

    const auto& entries = read_pool.entries();
    ASSERT_EQ(entries.size(), 1);

    const auto& entry = entries[0];
    EXPECT_EQ(entry.tag, 100);
    EXPECT_EQ(entry.length, test_text.size());

    const auto read_visitor = ov::util::VariantVisitor{[](const Bundle::BufferView&) {
                                                           FAIL() << "Expected stringstream*, got BufferView";
                                                       },
                                                       [&test_text](std::stringstream* ss) {
                                                           std::string read_text;
                                                           (*ss) >> read_text;
                                                           EXPECT_EQ(read_text, test_text);
                                                       },
                                                       [](const std::vector<Bundle::byte_t>&) {
                                                           FAIL() << "Expected stringstream*, got vector";
                                                       }};
    std::visit(read_visitor, entry.value);
}

}  // namespace ov::test
#endif  // BLOBS_STORAGE_ENABLED
