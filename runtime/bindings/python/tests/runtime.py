# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provide a layer of abstraction for an OpenVINO runtime environment."""

import logging
from typing import Dict, List, Union

import numpy as np

from openvino import Core

from openvino.exceptions import UserInputError
from openvino.impl import Function, Node, PartialShape, Type
from openvino.utils.types import NumericData, get_shape, get_dtype

import tests

log = logging.getLogger(__name__)


def runtime(backend_name: str = "CPU") -> "Runtime":
    """Create a Runtime object (helper factory)."""
    return Runtime(backend_name)


def get_runtime():
    """Return runtime object."""
    if tests.BACKEND_NAME is not None:
        return runtime(backend_name=tests.BACKEND_NAME)
    else:
        return runtime()


class Runtime(object):
    """Represents an nGraph runtime environment."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        log.debug("Creating Inference Engine for %s" % backend_name)
        self.backend = Core()
        assert backend_name in self.backend.available_devices, (
            'The requested device "' + backend_name + '" is not supported!'
        )

    def set_config(self, config: Dict[str, str]) -> None:
        """Set the inference engine configuration."""
        self.backend.set_config(config, device_name=self.backend_name)

    def __repr__(self) -> str:
        return "<Runtime: Backend='{}'>".format(self.backend_name)

    def computation(self, node_or_function: Union[Node, Function], *inputs: Node) -> "Computation":
        """Return a callable Computation object."""
        if isinstance(node_or_function, Node):
            ng_function = Function(node_or_function, inputs, node_or_function.name)
            return Computation(self, ng_function)
        elif isinstance(node_or_function, Function):
            return Computation(self, node_or_function)
        else:
            raise TypeError(
                "Runtime.computation must be called with an nGraph Function object "
                "or an nGraph node object an optionally Parameter node objects. "
                "Called with: %s",
                node_or_function,
            )


class Computation(object):
    """nGraph callable computation object."""

    def __init__(self, runtime: Runtime, ng_function: Function) -> None:
        self.runtime = runtime
        self.function = ng_function
        self.parameters = ng_function.get_parameters()
        self.results = ng_function.get_results()
        self.network_cache = {}

    def __repr__(self) -> str:
        params_string = ", ".join([param.name for param in self.parameters])
        return "<Computation: {}({})>".format(self.function.get_name(), params_string)

    def convert_buffers(self, source_buffers, target_dtypes):
        converted_buffers = []
        for i in range(len(source_buffers)):
            target_dtype = target_dtypes[i]
            # custom conversion for bf16
            if self.results[i].get_output_element_type(0) == Type.bf16:
                converted_buffers.append((source_buffers[i].view(np.uint32) >> 16).astype(np.uint16))
            else:
                converted_buffers.append(source_buffers[i].astype(target_dtype))
        return converted_buffers

    def __call__(self, *input_values: NumericData) -> List[NumericData]:
        """Run computation on input values and return result."""
        # Input validation
        if len(input_values) < len(self.parameters):
            raise UserInputError(
                "Expected %s params, received not enough %s values.", len(self.parameters), len(input_values)
            )

        param_types = [param.get_element_type() for param in self.parameters]
        param_names = [param.friendly_name for param in self.parameters]

        # ignore not needed input values
        input_values = [
            np.array(input_value[0], dtype=get_dtype(input_value[1]))
            for input_value in zip(input_values[: len(self.parameters)], param_types)
        ]
        input_shapes = [get_shape(input_value) for input_value in input_values]

        if self.network_cache.get(str(input_shapes)) is None:
            function = self.function
            if self.function.is_dynamic():
                function.reshape(dict(zip(param_names, [PartialShape(i) for i in input_shapes])))
            self.network_cache[str(input_shapes)] = function
        else:
            function = self.network_cache[str(input_shapes)]

        executable_network = self.runtime.backend.compile_model(function, self.runtime.backend_name)

        for parameter, input in zip(self.parameters, input_values):
            parameter_shape = parameter.get_output_partial_shape(0)
            input_shape = PartialShape(input.shape)
            if len(input.shape) > 0 and not parameter_shape.compatible(input_shape):
                raise UserInputError(
                    "Provided tensor's shape: %s does not match the expected: %s.",
                    input_shape,
                    parameter_shape,
                )

        request = executable_network.create_infer_request()
        result_buffers = request.infer(dict(zip(param_names, input_values)))
        # # Note: other methods to get result_buffers from request
        # # First call infer with no return value:
        # request.infer(dict(zip(param_names, input_values)))
        # # Now use any of following options:
        # result_buffers = [request.get_tensor(n).data for n in request.outputs]
        # result_buffers = [request.get_output_tensor(i).data for i in range(len(request.outputs))]
        # result_buffers = [t.data for t in request.output_tensors]

        # # Since OV overwrite result data type we have to convert results to the original one.
        original_dtypes = [get_dtype(result.get_output_element_type(0)) for result in self.results]
        converted_buffers = self.convert_buffers(result_buffers, original_dtypes)
        return converted_buffers
