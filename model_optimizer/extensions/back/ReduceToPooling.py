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

import numpy as np

from extensions.back.AvgPool import AvgPool
from extensions.back.MaxPool import MaxPool
from extensions.back.ScalarConstNormalize import ScalarNormalize
from mo.back.replacement import BackReplacementPattern
from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.pooling import Pooling
from mo.ops.power import AttributedPower
from mo.ops.reshape import Reshape


class ReduceReplacer(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    pool_method_map = {
        'ReduceMax': 'max',
        'ReduceMean': 'avg',
        'ReduceSum': 'avg',
    }

    supported_reduce_types = pool_method_map.keys()

    def run_before(self):
        from extensions.back.ReshapeMutation import ReshapeMutation
        return [ReshapeMutation, MaxPool, AvgPool, ScalarNormalize]

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', type=lambda node_type: node_type in self.supported_reduce_types))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['reduce']

        if node.out_port(0).data.get_value() is not None:
            # We leave Reduce* operations located in constant sub-graph as is
            # to keep model reshapable with --keep_shape_ops cli key
            return

        reduce_type = node.type
        if reduce_type not in self.pool_method_map:
            log.error("Reduce type {} is not included in pool_method_map. Please update pool_method_map with new key "
                      "{}".format(reduce_type, reduce_type))
            return

        input_data = node.in_node()
        output_data = node.out_node()

        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()

        # normalize node axes to exclude negative indices
        axes_data_value = node.in_port(1).data.get_value()
        axes = int64_array([axes_data_value.item()]) if axes_data_value.size == 1 else axes_data_value
        axes = [get_canonical_axis_index(input_shape, a) for a in axes]
        axes = sorted(axes)

        # Check that values in axes list are consecutive
        for idx in range(1, len(axes)):
            if axes[idx] != (axes[idx - 1] + 1):
                log.error("Reduce with not consecutive axes {} is not supported ".format(axes))
                return
        # So now we are sure that we can convert Reduce to appropriate operation

        # 1. Calculate shape that will be used in reduction
        reduction_dim = np.prod([input_shape[idx] for idx in axes])
        begin_dims = np.array([input_shape[idx] for idx in range(axes[0])])
        end_dim = np.prod([input_shape[idx] for idx in range(axes[-1] + 1, len(input_shape))])

        # 2. Create reshape with appropriate shape
        if len(begin_dims) > 2:
            if 0 not in axes:
                begin_dims = int64_array([begin_dims[0], np.prod(begin_dims[1:])])
            else:
                begin_dims = int64_array([np.prod(begin_dims[0:-1]), begin_dims[-1]])
        else:
            # Expand begin_dims to 2
            begin_dims = int64_array(np.append(begin_dims, [1] * (2 - len(begin_dims))))

        reshape_shape = np.array([*begin_dims, reduction_dim, end_dim], dtype=np.int64)
        pool_window = np.array([1, 1, reduction_dim, 1], dtype=np.int64)

        # 3. Reduce => Reshape->Pooling->Reshape
        reshape_op = Reshape(graph, {'name': node.id + '/Reshape'})
        reshape_dim_const_data = Const(graph, {'name': node.id + '/Reshape/Dim',
                                               'value': reshape_shape}).create_node_with_data()

        final_reshape_op = Reshape(graph, {'name': node.id + '/FinalReshape'})
        final_reshape_dim_const_data = Const(graph, {'name': node.id + '/FinalReshape/Dim',
                                                     'value': output_shape}).create_node_with_data()
        pooling_op = Pooling(graph,
                             dict(name=node.id + '/Pool',
                                  window=pool_window,
                                  output_spatial_shape=None,
                                  batch_dims=int64_array([0]),
                                  channel_dims=int64_array([1]),
                                  exclude_pad='false', pool_method=self.pool_method_map[reduce_type]))

        graph.remove_edge(input_data.id, node.id)
        graph.remove_edge(node.id, output_data.id)

        final_reshape_op.create_node_with_data(
            inputs=[pooling_op.create_node_with_data(
                inputs=[reshape_op.create_node_with_data(
                    inputs=[input_data, reshape_dim_const_data]
                )]
            ), final_reshape_dim_const_data],
            data_nodes=output_data)

        # convert batch dimension to 0 to produce reshape-able IR over the batch dimension
        if 0 not in axes:
            reshape_dim_const_data.in_node(0).value[0] = 0
            final_reshape_dim_const_data.in_node(0).value[0] = 0

        # 4. If it is reduction with summation, we need to multiply by size of the reduction slice with Mul op
        if reduce_type == 'ReduceSum':
            output_data.in_node().insert_node_with_data_after(
                output_data,
                AttributedPower,
                {'name': node.name + '/Mul', 'scale': float(reduction_dim)}
            )


class ReduceLogicalReplacer(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', type=lambda node_type: node_type is not None and
                                                                  node_type.startswith('ReduceLogical')))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['reduce']
        node.type = node.type.replace('Logical', '')
