"""
 Copyright (C) 2017-2020 Intel Corporation

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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error


class DepthToSpaceOp(Op):
    op = 'DepthToSpace'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,

            'mode': 'blocks_first',

            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        if self.ir_version == 10:
            return ['mode', 'block_size']
        else:
            return []

    @staticmethod
    def infer(node: Node):
        in_shape = node.in_node().shape
        if in_shape.size != 4:
            raise Error('TensorFlow DepthToSpace operation is supported for 4D \'NHWC\' input layout only. '
                        'Current input shape is \'{}\''.format(in_shape))
        N, H, W, C = in_shape
        block_size = node['block_size']
        if C % (block_size ** 2):
            raise Error('Feature dimensions of input tensor of DepthToSpace operation have to be divisible by square '
                        'of DepthToSpace \'block_size\' parameter. Input tensor shape = {}. Feature dimension = {}. '
                        'block_size = {}'.format(in_shape, C, block_size))
        out_shape = [N, int(H * block_size), int(W * block_size), int(C / (block_size ** 2))]
        if np.prod(in_shape) != np.prod(out_shape):
            return
        node.out_node().shape = int64_array(out_shape)
