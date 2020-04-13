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
import numpy as np

from extensions.ops.MatMul import FullyConnected
from extensions.ops.activation_ops import Tanh, Sigmoid
from extensions.ops.elementwise import Add, Mul
from extensions.ops.split import Split
from mo.front.caffe.extractors.utils import input_as_const
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.clamp import Clamp
from mo.ops.const import Const
from mo.ops.memory import Memory
from mo.ops.result import Result
from mo.ops.scale_shift import ScaleShiftOp


def unique_id(prefix: str = 'id') -> str:
    """
    Generates a unique id
    The optional string prefix can be specified.
    """
    index = len(unique_id.names)
    name = prefix
    while name in unique_id.names:
        name = '{}_{}'.format(prefix, index)
        index += 1
    unique_id.names.append(name)
    return name


unique_id.names = []


class ReplaceLSTMNodePattern(FrontReplacementOp):
    op = "LSTMCell"
    enabled = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        # current pass should be rewritten to use MatMul ops only (No FullyConnected ops should be created here)
        from extensions.front.MatMul_normalizer import FullyConnectedDecomposer
        from extensions.front.MoveEmbeddedInputsToInputs import MoveEmbeddedInputsToInputs
        return [FullyConnectedDecomposer,
                MoveEmbeddedInputsToInputs]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op=self.__class__.op, format='kaldi'))],
            edges=[]
        )

    def replace_op(self, graph: Graph, node: Node):
        input_node = node.in_node()

        memory_pair_input = unique_id('id')
        memory_pair_output = unique_id('id')

        # Input -> FullyConnected
        fc_layer_after_input_attrs = {'name': 'input_fullyconnected',
                                      'out-size': node.gifo_x_weights_shape[0],
                                      'transpose_weights': True,
                                      'bias_term': True,
                                      }

        fc_layer_after_input = FullyConnected(graph, fc_layer_after_input_attrs).create_node([input_node])
        input_as_const(fc_layer_after_input, fc_layer_after_input_attrs, 1, 'weights', node.gifo_x_weights)
        input_as_const(fc_layer_after_input, fc_layer_after_input_attrs, 2, 'biases', node.gifo_biases)

        prev_lstm_output = Memory(graph, {'name': 'prev_memory_output',
                                          'id': memory_pair_input,
                                          'index': 1,
                                          'size': 2,
                                          'shape': np.array([node.gifo_r_weights_shape[1]], dtype=np.int64)
                                          }).create_node()

        # *Memory(output) -> FullyConnected
        fc_layer_from_prev_state_attrs = {'name': 'prev_memory_output_fullyconnected',
                                          'out-size': node.gifo_r_weights_shape[0],
                                          'transpose_weights': True,
                                          'bias_term': False,
                                          }

        fc_layer_from_prev_state = FullyConnected(graph, fc_layer_from_prev_state_attrs).create_node(
            [prev_lstm_output])
        input_as_const(fc_layer_from_prev_state, fc_layer_from_prev_state_attrs, 1, 'weights', node.gifo_r_weights)

        # Memory -> FullyConnected  \
        #                           *Eltwise(sum)
        # Input -> FullyConnected   /
        join_input_prev_state_sum = Add(graph, {'name': 'join_input_eltwise',
                                                }).create_node([fc_layer_from_prev_state, fc_layer_after_input])

        # *Eltwise(sum) -> Split
        # it is split into 4 nodes: Act, Eltw*3
        # the following order is mandatory
        #       ___Tanh
        #      /
        # Split ---(2)Eltwise(sum)
        #     |\
        #     | \__(3)Eltwise(sum)
        #     |____(4)Eltwise(sum)
        split_joined_input_axis = Const(graph, {'value': np.int64(1)}).create_node()
        split_joined_input = Split(graph, {'name': 'join_input_split',
                                           'num_splits': 4,
                                           }).create_node([join_input_prev_state_sum, split_joined_input_axis])

        prev_lstm_state = Memory(graph, {'name': 'prev_memory_state',
                                         'id': memory_pair_output,
                                         'index': 1,
                                         'size': 2,
                                         'shape': np.array([node.input_gate_weights.shape[0]], dtype=np.int64)
                                         }).create_node()

        # *Memory(state) -> *ScaleShift(input)
        state_input_scaleshift_attrs = {'name': 'input_scaleshift',
                                        'bias_term': False
                                        }
        state_input_scaleshift = ScaleShiftOp(graph, state_input_scaleshift_attrs).create_node([prev_lstm_state])
        input_as_const(state_input_scaleshift, state_input_scaleshift_attrs, 1, 'weights', node.input_gate_weights)

        # *Memory(state) -> *ScaleShift(forget)
        state_forget_scaleshift_attrs = {'name': 'forget_scaleshift',
                                         'bias_term': False
                                         }
        state_forget_scaleshift = ScaleShiftOp(graph, state_forget_scaleshift_attrs).create_node([prev_lstm_state])
        input_as_const(state_forget_scaleshift, state_forget_scaleshift_attrs, 1, 'weights', node.forget_gate_weights)

        # Split                                 \
        #                                       (2)Eltwise(sum)
        # Memory(state) -> *ScaleShift(input)  /
        join_prev_lstm_input_joined_input_sum = Add(graph, {'name': 'join_prev_lstm_input_joined_input_eltwise',
                                                            }).create_node([(split_joined_input, 1),
                                                                            state_input_scaleshift
                                                                            ])
        # Split                                 \
        #                                       (3)Eltwise(sum)
        # Memory(state) -> *ScaleShift(forget)  /
        join_prev_lstm_input_joined_forget_sum = Add(graph, {'name': 'join_prev_lstm_input_joined_forget_sum',
                                                             }).create_node([(split_joined_input, 2),
                                                                             state_forget_scaleshift
                                                                             ])

        # Split -> Tanh
        remember_tahn = Tanh(graph, {'name': 'remember_tahnv'}).create_node([(split_joined_input, 0)])

        # Split -> (2)Eltwise(sum) -> *Sigmoid
        remember_sigmoid = Sigmoid(graph, {'name': 'remember_sigmoid'
                                           }).create_node([join_prev_lstm_input_joined_input_sum])

        # Split -> (3)Eltwise(sum) -> **Sigmoid
        forget_sigmoid = Sigmoid(graph, {'name': 'forget_sigmoid'
                                         }).create_node([join_prev_lstm_input_joined_forget_sum])

        # *Memory(state)                        \
        #                                       (6)Eltwise(mul)
        # Split -> (3)Eltwise(sum) -> **Sigmoid /
        join_forget_prev_state_mul = Mul(graph, {'name': 'join_forget_prev_state_mul',
                                                 }).create_node([forget_sigmoid, prev_lstm_state])

        # Split -> Tahn                         \
        #                                       (5)Eltwise(mul)
        # Split -> (2)Eltwise(sum) -> *Sigmoid   /
        join_remember_candidates_mul = Mul(graph, {'name': 'join_remember_candidates_mul',
                                                   }).create_node([remember_tahn, remember_sigmoid])

        # (5)Eltwise(mul)  \
        #               (7)Eltwise(sum)
        # (6)Eltwise(mul)   /
        join_forget_remember_sum = Add(graph, {'name': 'join_forget_remember_sum',
                                               }).create_node(
            [join_forget_prev_state_mul, join_remember_candidates_mul])

        # (7)Eltwise(sum) -> Clamp
        join_forget_clamp = Clamp(graph, {'name': 'join_forget_clamp',
                                          'max': node.clip_value,
                                          'min': -node.clip_value
                                          }).create_node(
            [join_forget_remember_sum])
        #
        # Clamp -> (2)Memory(state)
        next_lstm_state = Memory(graph, {'name': 'next_lstm_state',
                                         'id': memory_pair_output,
                                         'index': 0,
                                         'size': 2,
                                         'shape': np.array([node.input_gate_weights.shape[0]], dtype=np.int64)
                                         }).create_node([join_forget_clamp])
        Result(graph, {'name': 'next_lstm_state_out'}).create_node([next_lstm_state])

        # Clamp -> (2)Tahn
        state_filtered_tahn = Tanh(graph, {'name': 'state_filtered_tahn'}).create_node([join_forget_clamp])

        # Clamp -> (2)ScaleShift
        clamp_scaleshift_attrs = {'name': 'clamp_scaleshift',
                                  'bias_term': False}
        clamp_scaleshift = ScaleShiftOp(graph, clamp_scaleshift_attrs).create_node([join_forget_clamp])
        input_as_const(clamp_scaleshift, clamp_scaleshift_attrs, 1, 'weights', node.output_gate_weights)

        # Split                 \
        #                       (4)Eltwise(sum)
        # Clamp -> (2)ScaleShift /
        join_next_lstm_input_joined_input_sum = Add(graph, {'name': 'join_next_lstm_input_joined_input_sum',
                                                            }).create_node([(split_joined_input, 3), clamp_scaleshift])

        # (4)Eltwise(sum) -> (3)Sigmoid
        output_sigmoid = Sigmoid(graph, {'name': 'output_sigmoid'}).create_node([join_next_lstm_input_joined_input_sum])

        # (4)Eltwise(sum) -> (3)Sigmoid         \
        #                                       (5)Eltwise(mul)
        # Clamp -> (2)Tahn                      /
        joined_output_mul = Mul(graph, {'name': 'joined_output_mul'}).create_node([state_filtered_tahn, output_sigmoid])

        # (5)Eltwise(mul) -> (3)FullyConnected
        fc_output_attrs = {'name': 'FullyConnected',
                           'out-size': node.projection_weights_shape[0],
                           'transpose_weights': True,
                           'bias_term': False}
        fc_output = FullyConnected(graph, fc_output_attrs).create_node([joined_output_mul])
        input_as_const(fc_output, fc_output_attrs, 1, 'weights', node.projection_weights)

        #                   / (2)Memory(output)
        # (3)FullyConnected
        #                   \ Output (any next node) (edge created automatically after replacement)
        next_lstm_output = Memory(graph, {'name': 'next_lstm_output',
                                          'id': memory_pair_input,
                                          'index': 0,
                                          'size': 2,
                                          'shape': np.array([node.gifo_r_weights_shape[1]], dtype=np.int64)
                                          }).create_node([fc_output])
        Result(graph, {'name': 'next_lstm_output_out'}).create_node([next_lstm_output])

        return [fc_output.id]
