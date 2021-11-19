# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import datetime
import time

import openvino.opset8 as ops
from openvino import Core, AsyncInferQueue, Tensor, ProfilingInfo, Function

from ..conftest import model_path, read_image

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def create_function_with_memory(input_shape, data_type):
    input_data = ops.parameter(input_shape, name="input_data", dtype=data_type)
    rv = ops.read_value(input_data, "var_id_667")
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    func = Function(results=[res], sinks=[node], parameters=[input_data], name="name")
    return func


def test_get_profiling_info(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()
    request.infer({0: img})
    assert request.latency > 0
    prof_info = request.get_profiling_info()
    soft_max_node = next(node for node in prof_info if node.node_name == "fc_out")
    assert soft_max_node.node_type == "Softmax"
    assert soft_max_node.status == ProfilingInfo.Status.OPTIMIZED_OUT
    assert isinstance(soft_max_node.real_time, datetime.timedelta)
    assert isinstance(soft_max_node.cpu_time, datetime.timedelta)
    assert isinstance(soft_max_node.exec_type, str)


def test_tensor_setter(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net_1 = core.compile_model(model=func, device_name=device)
    exec_net_2 = core.compile_model(model=func, device_name=device)

    img = read_image()
    tensor = Tensor(img)

    request1 = exec_net_1.create_infer_request()
    request1.set_tensor("data", tensor)
    t1 = request1.get_tensor("data")

    assert np.allclose(tensor.data, t1.data, atol=1e-2, rtol=1e-2)

    res = request1.infer({0: tensor})
    res_1 = np.sort(res[0])
    t2 = request1.get_tensor("fc_out")
    assert np.allclose(t2.data, res[0].data, atol=1e-2, rtol=1e-2)

    request = exec_net_2.create_infer_request()
    res = request.infer({"data": tensor})
    res_2 = np.sort(request.get_tensor("fc_out").data)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)

    request.set_tensor("data", tensor)
    t3 = request.get_tensor("data")
    assert np.allclose(t3.data, t1.data, atol=1e-2, rtol=1e-2)


def test_set_tensors(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)

    data1 = read_image()
    tensor1 = Tensor(data1)
    data2 = np.ones(shape=(1, 10), dtype=np.float32)
    tensor2 = Tensor(data2)
    data3 = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor3 = Tensor(data3)
    data4 = np.zeros(shape=(1, 10), dtype=np.float32)
    tensor4 = Tensor(data4)

    request = exec_net.create_infer_request()
    request.set_tensors({"data": tensor1, "fc_out": tensor2})
    t1 = request.get_tensor("data")
    t2 = request.get_tensor("fc_out")
    assert np.allclose(tensor1.data, t1.data, atol=1e-2, rtol=1e-2)
    assert np.allclose(tensor2.data, t2.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensors({0: tensor2})
    output_node = exec_net.outputs[0]
    t3 = request.get_tensor(output_node)
    assert np.allclose(tensor2.data, t3.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensors({0: tensor1})
    output_node = exec_net.inputs[0]
    t4 = request.get_tensor(output_node)
    assert np.allclose(tensor1.data, t4.data, atol=1e-2, rtol=1e-2)

    output_node = exec_net.inputs[0]
    request.set_tensor(output_node, tensor3)
    t5 = request.get_tensor(output_node)
    assert np.allclose(tensor3.data, t5.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(tensor3)
    t6 = request.get_tensor(request.inputs[0])
    assert np.allclose(tensor3.data, t6.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(0, tensor1)
    t7 = request.get_tensor(request.inputs[0])
    assert np.allclose(tensor1.data, t7.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(tensor2)
    t8 = request.get_tensor(request.outputs[0])
    assert np.allclose(tensor2.data, t8.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(0, tensor4)
    t9 = request.get_tensor(request.outputs[0])
    assert np.allclose(tensor4.data, t9.data, atol=1e-2, rtol=1e-2)


def test_cancel(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()

    request.start_async({0: img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait()
    assert "[ INFER_CANCELLED ]" in str(e.value)

    request.start_async({"data": img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait_for(1)
    assert "[ INFER_CANCELLED ]" in str(e.value)


def test_start_async(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)
    img = read_image()
    jobs = 3
    requests = []
    for _ in range(jobs):
        requests.append(exec_net.create_infer_request())

    def callback(callbacks_info):
        time.sleep(0.01)
        callbacks_info["finished"] += 1

    callbacks_info = {}
    callbacks_info["finished"] = 0
    for request in requests:
        request.set_callback(callback, callbacks_info)
        request.start_async({0: img})
    for request in requests:
        request.wait()
        assert request.latency > 0
    assert callbacks_info["finished"] == jobs


def test_infer_mixed_keys(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    core.set_config({"PERF_COUNT": "YES"}, device)
    model = core.compile_model(func, device)

    img = read_image()
    tensor = Tensor(img)

    data2 = np.ones(shape=img.shape, dtype=np.float32)
    tensor2 = Tensor(data2)

    request = model.create_infer_request()
    res = request.infer({0: tensor2, "data": tensor})
    assert np.argmax(res) == 2


def test_infer_queue(device):
    jobs = 8
    num_request = 4
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)
    infer_queue = AsyncInferQueue(exec_net, num_request)
    jobs_done = [{"finished": False, "latency": 0} for _ in range(jobs)]

    def callback(request, job_id):
        jobs_done[job_id]["finished"] = True
        jobs_done[job_id]["latency"] = request.latency

    img = read_image()
    infer_queue.set_callback(callback)
    assert infer_queue.is_ready
    for i in range(jobs):
        infer_queue.start_async({"data": img}, i)
    infer_queue.wait_all()
    assert all(job["finished"] for job in jobs_done)
    assert all(job["latency"] > 0 for job in jobs_done)


@pytest.mark.parametrize("data_type",
                         [np.float32,
                          np.int32,
                          pytest.param(np.float16,
                                       marks=pytest.mark.xfail(reason="FP16 isn't "
                                                                      "supported in the CPU plugin"))])
@pytest.mark.parametrize("mode", ["set_init_memory_state", "reset_memory_state", "normal"])
@pytest.mark.parametrize("input_shape", [[10], [10, 10], [10, 10, 10], [2, 10, 10, 10]])
@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
                           "Memory layers fully supported only on CPU")
def test_query_state_write_buffer(device, input_shape, data_type, mode):
    core = Core()
    if device == "CPU":
        if core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
            pytest.skip("Can't run on ARM plugin")

    from openvino import Tensor
    from openvino.utils.types import get_dtype

    function = create_function_with_memory(input_shape, data_type)
    exec_net = core.compile_model(model=function, device_name=device)
    request = exec_net.create_infer_request()
    mem_states = request.query_state()
    mem_state = mem_states[0]

    assert mem_state.name == "var_id_667"
    # todo: Uncomment after fix 45611,
    #  CPU plugin returns outputs and memory state in FP32 in case of FP16 original precision
    # assert mem_state.state.tensor_desc.precision == data_type

    for i in range(1, 10):
        if mode == "set_init_memory_state":
            # create initial value
            const_init = 5
            init_array = np.full(input_shape, const_init, dtype=get_dtype(mem_state.state.element_type))
            tensor = Tensor(init_array)
            mem_state.state = tensor

            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, 1 + const_init, dtype=data_type)
        elif mode == "reset_memory_state":
            # reset initial state of ReadValue to zero
            mem_state.reset()
            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})

            # always ones
            expected_res = np.full(input_shape, 1, dtype=data_type)
        else:
            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, i, dtype=data_type)
        assert np.allclose(res[0], expected_res, atol=1e-6), \
            "Expected values: {} \n Actual values: {} \n".format(expected_res, res)
