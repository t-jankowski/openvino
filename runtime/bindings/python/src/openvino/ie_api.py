# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from typing import List, Union

from openvino.pyopenvino import TBlobFloat32
from openvino.pyopenvino import TBlobFloat64
from openvino.pyopenvino import TBlobInt64
from openvino.pyopenvino import TBlobUint64
from openvino.pyopenvino import TBlobInt32
from openvino.pyopenvino import TBlobUint32
from openvino.pyopenvino import TBlobInt16
from openvino.pyopenvino import TBlobUint16
from openvino.pyopenvino import TBlobInt8
from openvino.pyopenvino import TBlobUint8
from openvino.pyopenvino import TensorDesc
from openvino.pyopenvino import InferRequest
from openvino.pyopenvino import AsyncInferQueue
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Tensor


precision_map = {"FP32": np.float32,
                 "FP64": np.float64,
                 "FP16": np.int16,
                 "BF16": np.int16,
                 "I16": np.int16,
                 "I8": np.int8,
                 "BIN": np.int8,
                 "I32": np.int32,
                 "I64": np.int64,
                 "U8": np.uint8,
                 "BOOL": np.uint8,
                 "U16": np.uint16,
                 "U32": np.uint32,
                 "U64": np.uint64}


def normalize_inputs(py_dict: dict) -> dict:
    """Normalize a dictionary of inputs to contiguous numpy arrays."""
    return {k: (Tensor(v) if isinstance(v, np.ndarray) else v)
            for k, v in py_dict.items()}

# flake8: noqa: D102
def infer(request: InferRequest, inputs: dict = {}) -> np.ndarray:
    res = request._infer(inputs=normalize_inputs(inputs))
    # Required to return list since np.ndarray forces all of tensors data to match in
    # dimensions. This results in errors when running ops like variadic split.
    return [copy.deepcopy(tensor.data) for tensor in res]


def infer_new_request(exec_net: ExecutableNetwork, inputs: dict = None) -> List[np.ndarray]:
    res = exec_net._infer_new_request(inputs=normalize_inputs(inputs if inputs is not None else {}))
    # Required to return list since np.ndarray forces all of tensors data to match in
    # dimensions. This results in errors when running ops like variadic split.
    return [copy.deepcopy(tensor.data) for tensor in res]

# flake8: noqa: D102
def start_async(request: Union[InferRequest, AsyncInferQueue], inputs: dict = {}, userdata: dict = None) -> None:  # type: ignore
    request._start_async(inputs=normalize_inputs(inputs), userdata=userdata)

# flake8: noqa: C901
# Dispatch Blob types on Python side.
class BlobWrapper:
    def __new__(cls, tensor_desc: TensorDesc, arr: np.ndarray = None):  # type: ignore
        arr_size = 0
        precision = ""
        if tensor_desc is not None:
            tensor_desc_size = int(np.prod(tensor_desc.dims))
            precision = tensor_desc.precision
            if arr is not None:
                arr = np.array(arr)  # Keeping array as numpy array
                arr_size = int(np.prod(arr.shape))
                if np.isfortran(arr):
                    arr = arr.ravel(order="F")
                else:
                    arr = arr.ravel(order="C")
                if arr_size != tensor_desc_size:
                    raise AttributeError(f"Number of elements in provided numpy array "
                                         f"{arr_size} and required by TensorDesc "
                                         f"{tensor_desc_size} are not equal")
                if arr.dtype != precision_map[precision]:
                    raise ValueError(f"Data type {arr.dtype} of provided numpy array "
                                     f"doesn't match to TensorDesc precision {precision}")
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
            elif arr is None:
                arr = np.empty(0, dtype=precision_map[precision])
        else:
            raise AttributeError("TensorDesc can't be None")

        if precision in ["FP32"]:
            return TBlobFloat32(tensor_desc, arr, arr_size)
        elif precision in ["FP64"]:
            return TBlobFloat64(tensor_desc, arr, arr_size)
        elif precision in ["FP16", "BF16"]:
            return TBlobInt16(tensor_desc, arr.view(dtype=np.int16), arr_size)
        elif precision in ["I64"]:
            return TBlobInt64(tensor_desc, arr, arr_size)
        elif precision in ["U64"]:
            return TBlobUint64(tensor_desc, arr, arr_size)
        elif precision in ["I32"]:
            return TBlobInt32(tensor_desc, arr, arr_size)
        elif precision in ["U32"]:
            return TBlobUint32(tensor_desc, arr, arr_size)
        elif precision in ["I16"]:
            return TBlobInt16(tensor_desc, arr, arr_size)
        elif precision in ["U16"]:
            return TBlobUint16(tensor_desc, arr, arr_size)
        elif precision in ["I8", "BIN"]:
            return TBlobInt8(tensor_desc, arr, arr_size)
        elif precision in ["U8", "BOOL"]:
            return TBlobUint8(tensor_desc, arr, arr_size)
        else:
            raise AttributeError(f"Unsupported precision {precision} for Blob")

# flake8: noqa: D102
def blob_from_file(path_to_bin_file: str) -> BlobWrapper:
    array = np.fromfile(path_to_bin_file, dtype=np.uint8)
    tensor_desc = TensorDesc("U8", array.shape, "C")
    return BlobWrapper(tensor_desc, array)

# flake8: noqa: D102
def tensor_from_file(path: str) -> Tensor:
    """The data will be read with dtype of unit8"""
    return Tensor(np.fromfile(path, dtype=np.uint8))
