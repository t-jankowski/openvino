# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node
from mo.utils.runtime_info import OldAPIMapElementType


class ChangePlaceholderTypes(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    @staticmethod
    def is_node_casts_to_float_or_shapeof(node: Node):
        return (node.soft_get('type') == 'Convert' and node.soft_get('dst_type') == np.float32) or \
                node.soft_get('type') == 'ShapeOf'

    @staticmethod
    def update_type(node: Node, new_type: np.array):
        assert node.has_valid('rt_info')
        old_api_map = OldAPIMapElementType(version=0)
        attr_name = old_api_map.get_name()
        if (attr_name, old_api_map.get_version()) not in node.rt_info.info:
            node.rt_info.info[(attr_name, old_api_map.get_version())] = old_api_map
        node.rt_info.info[(attr_name, old_api_map.get_version())].set_legacy_type(new_type)

    def find_and_replace_pattern(self, graph: Graph):
        for op in graph.get_op_nodes(type='Parameter'):
            consumer_nodes = [p.node for p in op.out_port(0).get_destinations()]
            if all([ChangePlaceholderTypes.is_node_casts_to_float_or_shapeof(consumer) for consumer in consumer_nodes]):
                self.update_type(op, np.float32)

            if op.soft_get('data_type') == np.int64:
                self.update_type(op, np.int32)

            if op.soft_get('data_type') == np.uint8:
                self.update_type(op, np.float32)
