# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from openvino import Core, Tensor
from openvino.impl import Function


def np_dtype_to_tensor_type(data_type: np.dtype) -> int:
    """Return TensorProto type for provided numpy dtype.

    :param data_type: Numpy data type object.
    :return: TensorProto.DataType enum value for corresponding type.
    """
    return NP_TYPE_TO_TENSOR_TYPE[data_type]


def import_onnx_model(model: onnx.ModelProto) -> Function:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()
    ie = Core()
    func = ie.read_model(bytes(model_byte_string), Tensor(type=np.uint8, shape=[]))

    return func
