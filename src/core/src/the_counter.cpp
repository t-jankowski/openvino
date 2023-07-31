// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/the_counter.hpp"

#include <iostream>

namespace tj {
std::map<const char*, counter_base::counts_t> counter_base::counts;

counter_base::~counter_base() {
    count.dtor++;
    if (count.dtor == count.ctor + count.cctor + count.mctor)
        std::cout << "THE COUNT of '" << name << "':\n  dtor: " << count.dtor << "\n  ctor: " << count.ctor
                  << "\n  copy: " << count.cctor << "\n  move: " << count.mctor /* << "\n  copy=: " << count.cassign
                  << "\n  move=: " << count.massign */
                  << "\n";
}

counter_base::counter_base(const char* const n) : name{n}, count{counts[n]} {
    count.ctor++;
}

counter_base::counter_base(const counter_base& other) : name{other.name}, count{counts[name]} {
    count.cctor++;
}

counter_base::counter_base(counter_base&& other) : name{other.name}, count{counts[name]} {
    count.mctor++;
}

counter_base& counter_base::operator=(const counter_base&) {
    count.cassign++;
    return *this;
}

counter_base& counter_base::operator=(counter_base&&) {
    count.massign++;
    return *this;
}
}  // namespace tj
