// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>

#include "openvino/core/core_visibility.hpp"

namespace tj {
class counter_base {
    struct counts_t {
        size_t ctor;
        size_t cctor;
        size_t mctor;
        size_t dtor;
        size_t cassign;
        size_t massign;
    };
    static std::map<const char*, counts_t> counts;

    const char* const name;
    counts_t& count;

public:
    OPENVINO_API ~counter_base();
    OPENVINO_API counter_base(const char*);
    OPENVINO_API counter_base(const counter_base&);
    OPENVINO_API counter_base(counter_base&&);
    OPENVINO_API counter_base& operator=(const counter_base&);
    OPENVINO_API counter_base& operator=(counter_base&&);
};

template <typename T>
class the_counter : private counter_base {
public:
    OPENVINO_API ~the_counter() = default;
    OPENVINO_API the_counter() : counter_base(typeid(T).name()){};
    OPENVINO_API the_counter(const the_counter&) = default;
    OPENVINO_API the_counter(the_counter&&) = default;
    OPENVINO_API the_counter& operator=(const the_counter&) = default;
    OPENVINO_API the_counter& operator=(the_counter&&) = default;
};
}  // namespace tj
