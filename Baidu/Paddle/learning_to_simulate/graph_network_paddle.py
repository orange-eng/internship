

import re
from typing import Callable
import paddle.fluid as fluid
import paddle
from paddle.fluid.layers import tensor
import pgl
import paddle.nn as  nn
from pgl import graph

Reducer = Callable[[paddle.Tensor,paddle.Tensor],paddle.Tensor]

"""Build an MLP"""
# append命令是将整个对象加在列表末尾；而extend命令是将新对象中的元素逐一加在列表的末尾。
def build_mlp(hidden_size: int, num_hidden_layers:int,output_size:int):
  FC_layers = []
  FC_layers.extend([
    nn.Linear(in_features=hidden_size,out_features=output_size),
    nn.ReLU()])
  for _ in range(num_hidden_layers - 1):
    FC_layers.extend([
      nn.Linear(in_features=output_size,out_features=output_size),
      nn.ReLU()])
  FC_layers.append(nn.Softmax())    # 最后一层使用softmax
  return nn.Sequential(*FC_layers)


class EncodeProcessDecode(object):
  """Encode-Process-Decode function approximator for learnable simulator"""
  def __init__(
    self,
    latent_size:int, 
    mlp_hidden_size:int,
    mlp_num_hidden_layers:int,
    num_message_passing_steps:int,
    output_size:int,
    name: str="EncodeProcessDecode"):
    """Inits the model.
    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of message passing steps.
      output_size: Output size of the decode node representations as required
        by the downstream update function.
      reducer: Reduction to be used when aggregating the edges in the nodes in
        the interaction network. This should be a callable whose signature
        matches tf.math.unsorted_segment_sum.
      name: Name of the model.
    """  
    super().__init__(name=name)
    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size

  def _build(self,input_graph:pgl.graph.Graph):
    """Forward pass of the learnable dynamics model"""
    # Encode the input_graph
    latent_graph_0 = self._encode(input_graph)
    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)
    # Decode from the last latent graph.
    return self._decode(latent_graph_m)

  # Encoder
  def _encode(self,input_graph:pgl.graph.Graph)->pgl.graph.Graph:
    """Encodes the input graph features into a latent graph"""
    # Encode the node and edge features.
    def build_mlp_with_layer_norm():
      mlp = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._latent_size)
      return nn.Sequential([mlp,nn.LayerNorm()])

    latent_graph_0 = input_graph.replace(
      node_feat = build_mlp_with_layer_norm(input_graph.node_feat),
      edge_feat = build_mlp_with_layer_norm(input_graph.edge_feat)  
    )
    return latent_graph_0
  
  def _process(self,latent_graph_0:pgl.graph.Graph)->pgl.graph.Graph:
    """Processes the latent graph woth several steps of message passing"""
    # Do "m" message passing steps in the latent graphs
    def build_mlp_with_layer_norm():
      mlp = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._latent_size)
      return nn.Sequential([mlp,nn.LayerNorm()])

    latent_graph_k = latent_graph_0
    for _ in range(self._num_message_passing_steps):
      latent_graph_k = latent_graph_k.replace(
        node_feat = build_mlp_with_layer_norm(latent_graph_k.node_feat),
        edge_feat = build_mlp_with_layer_norm(latent_graph_k.edge_feat)  
      )
    return latent_graph_k

  def _decode(self,latent_graph:pgl.graph.Graph) -> paddle.fluid.Tensor:
    """Decodes from the latent graph"""
    output = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._output_size)(latent_graph.node_feat)
    return output    # 输入是node特征

