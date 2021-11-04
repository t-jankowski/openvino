# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: change the module name according to the description in 69196
from openvino.offline_transformations_pybind import apply_moc_transformations, apply_pot_transformations, \
    apply_low_latency_transformation, apply_pruning_transformation, apply_make_stateful_transformation

from openvino import Function, PartialShape
import openvino as ov


def get_test_function():
    param = ov.opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ov.opset8.relu(param)
    res = ov.opset8.result(relu, name="result")
    return Function([res], [param], "test")


def test_moc_transformations():
    function = get_test_function()

    apply_moc_transformations(function, False)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pot_transformations():
    function = get_test_function()

    apply_pot_transformations(function, "GNA")

    assert function is not None
    assert len(function.get_ops()) == 3


def test_low_latency_transformation():
    function = get_test_function()

    apply_low_latency_transformation(function, True)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pruning_transformation():
    function = get_test_function()

    apply_pruning_transformation(function)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_make_stateful_transformations():
    function = get_test_function()

    apply_make_stateful_transformation(function, {"parameter": "result"})

    assert function is not None
    assert len(function.get_parameters()) == 0
    assert len(function.get_results()) == 0
