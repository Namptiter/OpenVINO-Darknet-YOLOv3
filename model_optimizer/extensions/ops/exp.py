"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import networkx as nx
import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class ExpOp(Op):
    op = 'Exp'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 1
        assert len(node.out_nodes()) == 1
        input_node = node.in_node()
        assert input_node.has_valid('shape')
        node.out_node().shape = input_node.shape.copy()
        if input_node.has_valid('value'):
            node.out_node().value = np.exp(input_node.value)
