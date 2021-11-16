// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend_manager.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_FrontEnd(py::module m) {
    py::class_<ov::frontend::FrontEnd, std::shared_ptr<ov::frontend::FrontEnd>> fem(m,
                                                                                    "FrontEnd",
                                                                                    py::dynamic_attr(),
                                                                                    py::module_local());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def(
        "load",
        [](ov::frontend::FrontEnd& self, const std::string& s) {
            return self.load(s);
        },
        py::arg("path"),
        R"(
                Loads an input model by specified model file path.

                Parameters
                ----------
                path : str
                    Main model file path.

                Returns
                ----------
                load : InputModel
                    Loaded input model.
             )");

    fem.def(
        "convert",
        static_cast<std::shared_ptr<ngraph::Function> (ov::frontend::FrontEnd::*)(ov::frontend::InputModel::Ptr) const>(
            &ov::frontend::FrontEnd::convert),
        py::arg("model"),
        R"(
                Completely convert and normalize entire function, throws if it is not possible.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fem.def("convert",
            static_cast<void (ov::frontend::FrontEnd::*)(std::shared_ptr<ngraph::Function>) const>(
                &ov::frontend::FrontEnd::convert),
            py::arg("function"),
            R"(
                Completely convert the remaining, not converted part of a function.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fem.def("convert_partially",
            &ov::frontend::FrontEnd::convert_partially,
            py::arg("model"),
            R"(
                Convert only those parts of the model that can be converted leaving others as-is.
                Converted parts are not normalized by additional transformations; normalize function or
                another form of convert function should be called to finalize the conversion process.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert_partially : Function
                    Partially converted nGraph function.
             )");

    fem.def("decode",
            &ov::frontend::FrontEnd::decode,
            py::arg("model"),
            R"(
                Convert operations with one-to-one mapping with decoding nodes.
                Each decoding node is an nGraph node representing a single FW operation node with
                all attributes represented in FW-independent way.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                decode : Function
                    nGraph function after decoding.
             )");

    fem.def("normalize",
            &ov::frontend::FrontEnd::normalize,
            py::arg("function"),
            R"(
                Runs normalization passes on function that was loaded with partial conversion.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.
             )");

    fem.def("get_name",
            &ov::frontend::FrontEnd::get_name,
            R"(
                Gets name of this FrontEnd. Can be used by clients
                if frontend is selected automatically by FrontEndManager::load_by_model.

                Parameters
                ----------
                get_name : str
                    Current frontend name. Empty string if not implemented.
            )");

    fem.def("__repr__", [](const ov::frontend::FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}
