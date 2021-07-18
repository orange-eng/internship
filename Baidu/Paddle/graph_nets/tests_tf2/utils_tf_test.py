# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for utils_tf.py in Tensorflow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.tests import test_utils as test_utils_tf1
from graph_nets.tests_tf2 import test_utils
import networkx as nx
import numpy as np
from six.moves import range
import tensorflow as tf
import tree




class RepeatTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `repeat`."""

  @parameterized.named_parameters(
      ("base", (3,), [2, 3, 4], 0),
      ("empty_group_first", (3,), [0, 3, 4], 0),
      ("empty_group_middle", (3,), [2, 0, 4], 0),
      ("double_empty_group_middle", (4,), [2, 0, 0, 4], 0),
      ("empty_group_last", (3,), [2, 3, 0], 0),
      ("just_one_group", (1,), [2], 0),
      ("zero_groups", (0,), [], 0),
      ("axis 0", (2, 3, 4), [2, 3], 0),
      ("axis 1", (3, 2, 4), [2, 3], 1),
      ("axis 2", (4, 3, 2), [2, 3], 2),
      ("zero_groups_with_shape", (2, 0, 4), [], 1),
      )
  def test_repeat(self, shape, repeats, axis):
    num_elements = np.prod(shape)
    t = np.arange(num_elements).reshape(*shape)
    expected = np.repeat(t, repeats, axis=axis)
    tensor = tf.constant(t)
    repeats = tf.constant(repeats, dtype=tf.int32)
    actual = utils_tf.repeat(tensor, repeats, axis=axis)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(("default", "custom_name", None),
                                  ("custom", None, "repeat"))
  def test_name_scope(self, name, expected_name):
    self.skipTest("Uses get_default_graph.")
    kwargs = {"name": name} if name else {}
    expected_name = expected_name if expected_name else name

    t = tf.zeros([3, 2, 4])
    indices = tf.constant([2, 3])
    with test_utils.assert_new_op_prefixes(self, expected_name + "/"):
      utils_tf.repeat(t, indices, axis=1, **kwargs)


def _generate_graph(batch_index, n_nodes=4, add_edges=True):
  graph = nx.DiGraph()
  for node in range(n_nodes):
    node_data = {"features": np.array([node, batch_index], dtype=np.float32)}
    graph.add_node(node, **node_data)
  if add_edges:
    for edge, (receiver, sender) in enumerate(zip([0, 0, 1], [1, 2, 3])):
      if sender < n_nodes and receiver < n_nodes:
        edge_data = np.array([edge, edge + 1, batch_index], dtype=np.float64)
        graph.add_edge(sender, receiver, features=edge_data, index=edge)
  graph.graph["features"] = np.array([batch_index], dtype=np.float32)
  return graph


class ConcatTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `concat`, along various axis."""

  @parameterized.named_parameters(
      ("no nones", []), ("stateless graph", ["nodes", "edges", "globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_concat_first_axis(self, none_fields):
    graph_0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph_1 = utils_np.networkxs_to_graphs_tuple([_generate_graph(2, 2)])
    graph_2 = utils_np.networkxs_to_graphs_tuple([_generate_graph(3, 3)])
    graphs_ = [
        gr.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
        for gr in [graph_0, graph_1, graph_2]
    ]
    graphs_ = [gr.map(lambda _: None, none_fields) for gr in graphs_]
    concat_graph = utils_tf.concat(graphs_, axis=0)
    for none_field in none_fields:
      self.assertIsNone(getattr(concat_graph, none_field))
    concat_graph = concat_graph.map(tf.no_op, none_fields)
    if "nodes" not in none_fields:
      self.assertAllEqual(
          np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 2]),
          [x[0] for x in concat_graph.nodes])
      self.assertAllEqual(
          np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]),
          [x[1] for x in concat_graph.nodes])
    if "edges" not in none_fields:
      self.assertAllEqual(
          np.array([0, 1, 0, 0, 0, 1]), [x[0] for x in concat_graph.edges])
      self.assertAllEqual(
          np.array([0, 0, 1, 2, 3, 3]), [x[2] for x in concat_graph.edges])
    self.assertAllEqual(np.array([3, 2, 2, 3]), concat_graph.n_node)
    self.assertAllEqual(np.array([2, 1, 1, 2]), concat_graph.n_edge)
    if "senders" not in none_fields:
      # [1, 2], [1], [1], [1, 2] and 3, 2, 2, 3 nodes
      # So we are summing [1, 2, 1, 1, 2] with [0, 0, 3, 5, 7, 7]
      self.assertAllEqual(np.array([1, 2, 4, 6, 8, 9]), concat_graph.senders)
    if "receivers" not in none_fields:
      # [0, 0], [0], [0], [0, 0] and 3, 2, 2, 3 nodes
      # So we are summing [0, 0, 0, 0, 0, 0] with [0, 0, 3, 5, 7, 7]
      self.assertAllEqual(np.array([0, 0, 3, 5, 7, 7]), concat_graph.receivers)
    if "globals" not in none_fields:
      self.assertAllEqual(np.array([[0], [1], [2], [3]]), concat_graph.globals)

  def test_nested_features(self):
    graph_0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph_1 = utils_np.networkxs_to_graphs_tuple([_generate_graph(2, 2)])
    graph_2 = utils_np.networkxs_to_graphs_tuple([_generate_graph(3, 3)])
    graphs_ = [
        gr.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
        for gr in [graph_0, graph_1, graph_2]
    ]

    def _create_nested_fields(graphs_tuple):
      new_nodes = ({"a": graphs_tuple.nodes,
                    "b": [graphs_tuple.nodes + 1,
                          graphs_tuple.nodes + 2]
                    },)

      new_edges = [{"c": graphs_tuple.edges + 5,
                    "d": (graphs_tuple.edges + 1,
                          graphs_tuple.edges + 3),
                    }]
      new_globals = []

      return graphs_tuple.replace(nodes=new_nodes,
                                  edges=new_edges,
                                  globals=new_globals)

    graphs_ = [_create_nested_fields(gr) for gr in graphs_]
    concat_graph = utils_tf.concat(graphs_, axis=0)

    actual_nodes = concat_graph.nodes
    actual_edges = concat_graph.edges
    actual_globals = concat_graph.globals

    expected_nodes = tree.map_structure(
        lambda *x: tf.concat(x, axis=0), *[gr.nodes for gr in graphs_])
    expected_edges = tree.map_structure(
        lambda *x: tf.concat(x, axis=0), *[gr.edges for gr in graphs_])
    expected_globals = tree.map_structure(
        lambda *x: tf.concat(x, axis=0), *[gr.globals for gr in graphs_])

    tree.assert_same_structure(expected_nodes, actual_nodes)
    tree.assert_same_structure(expected_edges, actual_edges)
    tree.assert_same_structure(expected_globals, actual_globals)

    tree.map_structure(self.assertAllEqual, expected_nodes, actual_nodes)
    tree.map_structure(self.assertAllEqual, expected_edges, actual_edges)
    tree.map_structure(self.assertAllEqual, expected_globals, actual_globals)

    # Borrowed from `test_concat_first_axis`:
    self.assertAllEqual(np.array([3, 2, 2, 3]), concat_graph.n_node)
    self.assertAllEqual(np.array([2, 1, 1, 2]), concat_graph.n_edge)
    self.assertAllEqual(np.array([1, 2, 4, 6, 8, 9]), concat_graph.senders)
    self.assertAllEqual(np.array([0, 0, 3, 5, 7, 7]), concat_graph.receivers)

  def test_concat_last_axis(self):
    graph0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph1 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(2, 3), _generate_graph(3, 2)])
    graph0 = graph0.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
    graph1 = graph1.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
    concat_graph = utils_tf.concat([graph0, graph1], axis=-1)
    self.assertAllEqual(
        np.array([[0, 0, 0, 2], [1, 0, 1, 2], [2, 0, 2, 2], [0, 1, 0, 3],
                  [1, 1, 1, 3]]), concat_graph.nodes)
    self.assertAllEqual(
        np.array([[0, 1, 0, 0, 1, 2], [1, 2, 0, 1, 2, 2], [0, 1, 1, 0, 1, 3]]),
        concat_graph.edges)
    self.assertAllEqual(np.array([3, 2]), concat_graph.n_node)
    self.assertAllEqual(np.array([2, 1]), concat_graph.n_edge)
    self.assertAllEqual(np.array([1, 2, 4]), concat_graph.senders)
    self.assertAllEqual(np.array([0, 0, 3]), concat_graph.receivers)
    self.assertAllEqual(np.array([[0, 2], [1, 3]]), concat_graph.globals)

  @parameterized.parameters(
      ("nodes"),
      ("edges"),
      ("globals"),
      )
  def test_raise_all_or_no_nones(self, none_field):
    graph_0 = utils_np.networkxs_to_graphs_tuple(
        [_generate_graph(0, 3), _generate_graph(1, 2)])
    graph_1 = utils_np.networkxs_to_graphs_tuple([_generate_graph(2, 2)])
    graph_2 = utils_np.networkxs_to_graphs_tuple([_generate_graph(3, 3)])
    graphs_ = [
        gr.map(tf.convert_to_tensor, graphs.ALL_FIELDS)
        for gr in [graph_0, graph_1, graph_2]
    ]
    graphs_[1] = graphs_[1].replace(**{none_field: None})
    with self.assertRaisesRegex(
        ValueError,
        "Different set of keys found when iterating over data dictionaries."):
      utils_tf.concat(graphs_, axis=0)


class StopGradientsGraphTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(StopGradientsGraphTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.zeros([10], dtype=tf.int32),
        "receivers": tf.zeros([10], dtype=tf.int32),
        "nodes": tf.ones([5, 7]),
        "edges": tf.zeros([10, 6]),
        "globals": tf.zeros([1, 8])
    }])

  def _check_if_gradients_exist(self, stopped_gradients_graph):
    gradients = []
    for field in ["globals", "nodes", "edges"]:
      with tf.GradientTape() as tape:
        xs = getattr(self._graph, field)
        ys = getattr(stopped_gradients_graph, field)
      gradient = tape.gradient(ys, xs) if ys is not None else ys
      gradients.append(gradient)
    return [True if grad is not None else False for grad in gradients]

  @parameterized.named_parameters(
      ("stop_all_fields", True, True, True),
      ("stop_globals", True, False, False), ("stop_nodes", False, True, False),
      ("stop_edges", False, False, True), ("stop_none", False, False, False))
  def test_stop_gradients_outputs(self, stop_globals, stop_nodes, stop_edges):
    stopped_gradients_graph = utils_tf.stop_gradient(
        self._graph,
        stop_globals=stop_globals,
        stop_nodes=stop_nodes,
        stop_edges=stop_edges)

    gradients_exist = self._check_if_gradients_exist(stopped_gradients_graph)
    expected_gradients_exist = [
        not stop_globals, not stop_nodes, not stop_edges
    ]
    self.assertAllEqual(expected_gradients_exist, gradients_exist)

  @parameterized.named_parameters(("no_nodes", "nodes"), ("no_edges", "edges"),
                                  ("no_globals", "globals"))
  def test_stop_gradients_with_missing_field_raises(self, none_field):
    self._graph = self._graph.map(lambda _: None, [none_field])
    with self.assertRaisesRegex(ValueError, none_field):
      utils_tf.stop_gradient(self._graph)

  def test_stop_gradients_default_params(self):
    """Tests for the default params of `utils_tf.stop_gradient`."""
    stopped_gradients_graph = utils_tf.stop_gradient(self._graph)
    gradients_exist = self._check_if_gradients_exist(stopped_gradients_graph)
    expected_gradients_exist = [False, False, False]
    self.assertAllEqual(expected_gradients_exist, gradients_exist)


class IdentityTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the `identity` method."""

  def setUp(self):
    super(IdentityTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "receivers": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "nodes": tf.random.uniform([5, 7]),
        "edges": tf.random.uniform([10, 6]),
        "globals": tf.random.uniform([1, 8])
    }])

  def test_name_scope(self):
    """Tests that the name scope are correctly pushed through this function."""
    self.skipTest("Tensor.name is meaningless when eager execution is enabled")

  @parameterized.named_parameters(
      ("all fields defined", []), ("no node features", ["nodes"]),
      ("no edge features", ["edges"]), ("no global features", ["globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_output(self, none_fields):
    """Tests that this function produces the identity."""
    graph = self._graph.map(lambda _: None, none_fields)
    with tf.name_scope("test"):
      graph_id = utils_tf.identity(graph)
    expected_out = utils_tf.nest_to_numpy(graph)
    actual_out = utils_tf.nest_to_numpy(graph_id)
    for field in [
        "nodes", "edges", "globals", "receivers", "senders", "n_node", "n_edge"
    ]:
      if field in none_fields:
        self.assertIsNone(getattr(actual_out, field))
      else:
        self.assertNDArrayNear(
            getattr(expected_out, field), getattr(actual_out, field), err=1e-4)


class RunGraphWithNoneTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunGraphWithNoneTest, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "receivers": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "nodes": tf.random.uniform([5, 7]),
        "edges": tf.random.uniform([10, 6]),
        "globals": tf.random.uniform([1, 8])
    }])

  @parameterized.named_parameters(
      ("all fields defined", []), ("no node features", ["nodes"]),
      ("no edge features", ["edges"]), ("no global features", ["globals"]),
      ("no edges", ["edges", "receivers", "senders"]))
  def test_output(self, none_fields):
    """Tests that this function produces the identity."""
    graph_id = self._graph.map(lambda _: None, none_fields)
    graph = graph_id.map(tf.no_op, none_fields)
    expected_out = graph
    actual_out = graph_id
    for field in [
        "nodes", "edges", "globals", "receivers", "senders", "n_node", "n_edge"
    ]:
      if field in none_fields:
        self.assertIsNone(getattr(actual_out, field))
      else:
        self.assertNDArrayNear(
            getattr(expected_out, field), getattr(actual_out, field), err=1e-4)


class ComputeOffsetTest(tf.test.TestCase):
  """Tests for the `compute_stacked_offsets` method."""

  def setUp(self):
    super(ComputeOffsetTest, self).setUp()
    self.sizes = [5, 4, 3, 1, 2, 0, 3, 0, 4, 7]
    self.repeats = [2, 2, 0, 2, 1, 3, 2, 0, 3, 2]
    self.offset = [
        0, 0, 5, 5, 12, 12, 13, 15, 15, 15, 15, 15, 18, 18, 18, 22, 22
    ]

  def test_compute_stacked_offsets(self):
    offset0 = utils_tf._compute_stacked_offsets(
        self.sizes, self.repeats)
    offset1 = utils_tf._compute_stacked_offsets(
        np.array(self.sizes), np.array(self.repeats))
    offset2 = utils_tf._compute_stacked_offsets(
        tf.constant(self.sizes, dtype=tf.int32),
        tf.constant(self.repeats, dtype=tf.int32))

    self.assertAllEqual(self.offset, offset0.numpy().tolist())
    self.assertAllEqual(self.offset, offset1.numpy().tolist())
    self.assertAllEqual(self.offset, offset2.numpy().tolist())


class DataDictsCompletionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the methods creating complete graphs from partial graphs."""

  def _assert_indices_sizes(self, dict_, n_relation):
    for key in ["receivers", "senders"]:
      self.assertAllEqual((n_relation,), dict_[key].get_shape().as_list())

  @parameterized.named_parameters(
      ("static", utils_tf._create_complete_edges_from_nodes_static),
      ("dynamic", utils_tf._create_complete_edges_from_nodes_dynamic),
  )
  def test_create_complete_edges_from_nodes_include_self_edges(self, method):
    for graph_dict in self.graphs_dicts_in:
      n_node = graph_dict["nodes"].shape[0]
      edges_dict = method(n_node, exclude_self_edges=False)
      self._assert_indices_sizes(edges_dict, n_node**2)

  @parameterized.named_parameters(
      ("static", utils_tf._create_complete_edges_from_nodes_static),
      ("dynamic", utils_tf._create_complete_edges_from_nodes_dynamic),
  )
  def test_create_complete_edges_from_nodes_exclude_self_edges(self, method):
    for graph_dict in self.graphs_dicts_in:
      n_node = graph_dict["nodes"].shape[0]
      edges_dict = method(n_node, exclude_self_edges=True)
      self._assert_indices_sizes(edges_dict, n_node * (n_node - 1))

  def test_create_complete_edges_from_nodes_dynamic_number_of_nodes(self):
    for graph_dict in self.graphs_dicts_in:
      n_node = tf.shape(tf.constant(graph_dict["nodes"]))[0]
      edges_dict = utils_tf._create_complete_edges_from_nodes_dynamic(
          n_node, exclude_self_edges=False)
      n_relation = n_node**2
      receivers = edges_dict["receivers"].numpy()
      senders = edges_dict["senders"].numpy()
      n_edge = edges_dict["n_edge"].numpy()
      self.assertAllEqual((n_relation,), receivers.shape)
      self.assertAllEqual((n_relation,), senders.shape)
      self.assertEqual(n_relation, n_edge)


class GraphsCompletionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for completing partial GraphsTuple."""

  def _assert_indices_sizes(self, graph, n_relation):
    for key in ["receivers", "senders"]:
      self.assertAllEqual((n_relation,),
                          getattr(graph, key).get_shape().as_list())

  @parameterized.named_parameters(("edge size 0", 0), ("edge size 1", 1))
  def test_fill_edge_state(self, edge_size):
    """Tests for filling the edge state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_edges = np.sum(self.reference_graph.n_edge)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size)
    self.assertAllEqual((n_edges, edge_size),
                        graphs_tuple.edges.get_shape().as_list())

  @parameterized.named_parameters(("edge size 0", 0), ("edge size 1", 1))
  def test_fill_edge_state_dynamic(self, edge_size):
    """Tests for filling the edge state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple._replace(
        n_edge=tf.constant(
            graphs_tuple.n_edge, shape=graphs_tuple.n_edge.get_shape()))
    n_edges = np.sum(self.reference_graph.n_edge)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size)
    actual_edges = graphs_tuple.edges
    self.assertNDArrayNear(
        np.zeros((n_edges, edge_size)), actual_edges, err=1e-4)

  @parameterized.named_parameters(("global size 0", 0), ("global size 1", 1))
  def test_fill_global_state(self, global_size):
    """Tests for filling the global state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("globals")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_graphs = self.reference_graph.n_edge.shape[0]
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, global_size)
    self.assertAllEqual((n_graphs, global_size),
                        graphs_tuple.globals.get_shape().as_list())

  @parameterized.named_parameters(("global size 0", 0), ("global size 1", 1))
  def test_fill_global_state_dynamic(self, global_size):
    """Tests for filling the global state with a constant content."""
    for g in self.graphs_dicts_in:
      g.pop("globals")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    # Hide global shape information
    graphs_tuple = graphs_tuple._replace(
        n_node=tf.constant(
            graphs_tuple.n_node, shape=graphs_tuple.n_edge.get_shape()))
    n_graphs = self.reference_graph.n_edge.shape[0]
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, global_size)
    actual_globals = graphs_tuple.globals.numpy()
    self.assertNDArrayNear(
        np.zeros((n_graphs, global_size)), actual_globals, err=1e-4)

  @parameterized.named_parameters(("node size 0", 0), ("node size 1", 1))
  def test_fill_node_state(self, node_size):
    """Tests for filling the node state with a constant content."""
    for g in self.graphs_dicts_in:
      g["n_node"] = g["nodes"].shape[0]
      g.pop("nodes")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    n_nodes = np.sum(self.reference_graph.n_node)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size)
    self.assertAllEqual((n_nodes, node_size),
                        graphs_tuple.nodes.get_shape().as_list())

  @parameterized.named_parameters(("node size 0", 0), ("node size 1", 1))
  def test_fill_node_state_dynamic(self, node_size):
    """Tests for filling the node state with a constant content."""
    for g in self.graphs_dicts_in:
      g["n_node"] = g["nodes"].shape[0]
      g.pop("nodes")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple._replace(
        n_node=tf.constant(
            graphs_tuple.n_node, shape=graphs_tuple.n_node.get_shape()))
    n_nodes = np.sum(self.reference_graph.n_node)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size)
    actual_nodes = graphs_tuple.nodes.numpy()
    self.assertNDArrayNear(
        np.zeros((n_nodes, node_size)), actual_nodes, err=1e-4)

  def test_fill_edge_state_with_missing_fields_raises(self):
    """Edge field cannot be filled if receivers or senders are missing."""
    for g in self.graphs_dicts_in:
      g.pop("receivers")
      g.pop("senders")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    with self.assertRaisesRegex(ValueError, "receivers"):
      graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size=1)

  def test_fill_state_default_types(self):
    """Tests that the features are created with the correct default type."""
    for g in self.graphs_dicts_in:
      g.pop("nodes")
      g.pop("globals")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, edge_size=1)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, node_size=1)
    graphs_tuple = utils_tf.set_zero_global_features(
        graphs_tuple, global_size=1)
    self.assertEqual(tf.float32, graphs_tuple.edges.dtype)
    self.assertEqual(tf.float32, graphs_tuple.nodes.dtype)
    self.assertEqual(tf.float32, graphs_tuple.globals.dtype)

  @parameterized.parameters(
      (tf.float64,),
      (tf.int32,),
  )
  def test_fill_state_user_specified_types(self, dtype):
    """Tests that the features are created with the correct default type."""
    for g in self.graphs_dicts_in:
      g.pop("nodes")
      g.pop("globals")
      g.pop("edges")
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, 1, dtype)
    graphs_tuple = utils_tf.set_zero_node_features(graphs_tuple, 1, dtype)
    graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, 1, dtype)
    self.assertEqual(dtype, graphs_tuple.edges.dtype)
    self.assertEqual(dtype, graphs_tuple.nodes.dtype)
    self.assertEqual(dtype, graphs_tuple.globals.dtype)

  @parameterized.named_parameters(
      ("no self edges", False),
      ("self edges", True),
  )
  def test_fully_connect_graph_dynamic(self, exclude_self_edges):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    n_relation = 0
    for g in self.graphs_dicts_in:
      n_node = g["nodes"].shape[0]
      if exclude_self_edges:
        n_relation += n_node * (n_node - 1)
      else:
        n_relation += n_node * n_node

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = utils_tf.fully_connect_graph_dynamic(graphs_tuple,
                                                        exclude_self_edges)
    actual_receivers = graphs_tuple.receivers.numpy()
    actual_senders = graphs_tuple.senders.numpy()

    self.assertAllEqual((n_relation,), actual_receivers.shape)
    self.assertAllEqual((n_relation,), actual_senders.shape)
    self.assertAllEqual((len(self.graphs_dicts_in),),
                        graphs_tuple.n_edge.get_shape().as_list())

  @parameterized.named_parameters(
      ("no self edges", False),
      ("self edges", True),
  )
  def test_fully_connect_graph_dynamic_with_dynamic_sizes(
      self, exclude_self_edges):
    for g in self.graphs_dicts_in:
      g.pop("edges")
      g.pop("receivers")
      g.pop("senders")
    n_relation = 0
    for g in self.graphs_dicts_in:
      n_node = g["nodes"].shape[0]
      if exclude_self_edges:
        n_relation += n_node * (n_node - 1)
      else:
        n_relation += n_node * n_node

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs_tuple = graphs_tuple.map(test_utils.mask_leading_dimension,
                                    ["nodes", "globals", "n_node", "n_edge"])
    graphs_tuple = utils_tf.fully_connect_graph_dynamic(graphs_tuple,
                                                        exclude_self_edges)

    actual_receivers = graphs_tuple.receivers.numpy()
    actual_senders = graphs_tuple.senders.numpy()
    actual_n_edge = graphs_tuple.n_edge.numpy()
    self.assertAllEqual((n_relation,), actual_receivers.shape)
    self.assertAllEqual((n_relation,), actual_senders.shape)
    self.assertAllEqual((len(self.graphs_dicts_in),), actual_n_edge.shape)
    expected_edges = []
    offset = 0
    for graph in self.graphs_dicts_in:
      n_node = graph["nodes"].shape[0]
      for e1 in range(n_node):
        for e2 in range(n_node):
          if not exclude_self_edges or e1 != e2:
            expected_edges.append((e1 + offset, e2 + offset))
      offset += n_node
    actual_edges = zip(actual_receivers, actual_senders)
    self.assertSetEqual(set(actual_edges), set(expected_edges))


class GraphsTupleConversionTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the method converting between data dicts and GraphsTuple."""

  @parameterized.named_parameters(("all fields defined", []), (
      "no edge features",
      ["edges"],
  ), (
      "no node features",
      ["nodes"],
  ), (
      "no globals",
      ["globals"],
  ), (
      "no edges",
      ["edges", "receivers", "senders"],
  ))
  def test_data_dicts_to_graphs_tuple(self, none_fields):
    """Fields in `none_fields` will be cleared out."""
    for field in none_fields:
      for graph_dict in self.graphs_dicts_in:
        if field in graph_dict:
          if field == "nodes":
            graph_dict["n_node"] = graph_dict["nodes"].shape[0]
          graph_dict[field] = None
        self.reference_graph = self.reference_graph._replace(**{field: None})
      if field == "senders":
        self.reference_graph = self.reference_graph._replace(
            n_edge=np.zeros_like(self.reference_graph.n_edge))
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for field in none_fields:
      self.assertIsNone(getattr(graphs_tuple, field))
    graphs_tuple = graphs_tuple.map(tf.no_op, none_fields)
    self._assert_graph_equals_np(self.reference_graph, graphs_tuple)

  @parameterized.parameters(("receivers",), ("senders",))
  def test_data_dicts_to_graphs_tuple_raises(self, none_field):
    """Fields that cannot be missing."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict[none_field] = None
    with self.assertRaisesRegex(ValueError, none_field):
      utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)

  def test_data_dicts_to_graphs_tuple_no_raise(self):
    """Not having nodes is fine, if the number of nodes is provided."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = graph_dict["nodes"].shape[0]
      graph_dict["nodes"] = None
    utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)

  def test_data_dicts_to_graphs_tuple_cast_types(self):
    """Index and number fields should be cast to tensors of the right type."""
    for graph_dict in self.graphs_dicts_in:
      graph_dict["n_node"] = np.array(
          graph_dict["nodes"].shape[0], dtype=np.int64)
      graph_dict["receivers"] = graph_dict["receivers"].astype(np.int16)
      graph_dict["senders"] = graph_dict["senders"].astype(np.float64)
      graph_dict["nodes"] = graph_dict["nodes"].astype(np.float64)
      graph_dict["edges"] = tf.constant(graph_dict["edges"], dtype=tf.float64)
    out = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    for key in ["n_node", "n_edge", "receivers", "senders"]:
      self.assertEqual(tf.int32, getattr(out, key).dtype)
      self.assertEqual(type(tf.int32), type(getattr(out, key).dtype))
    for key in ["nodes", "edges"]:
      self.assertEqual(type(tf.float64), type(getattr(out, key).dtype))
      self.assertEqual(tf.float64, getattr(out, key).dtype)


class GraphsIndexingTests(test_utils.GraphsTest, parameterized.TestCase):
  """Tests for the `get_graph` method."""

  @parameterized.named_parameters(("int_index", False),
                                  ("tensor_index", True))
  def test_getitem_one(self, use_tensor_index):
    index = 2
    expected = self.graphs_dicts_out[index]

    if use_tensor_index:
      index = tf.constant(index)

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graph = utils_tf.get_graph(graphs_tuple, index)

    graph = utils_tf.nest_to_numpy(graph)
    actual, = utils_np.graphs_tuple_to_data_dicts(graph)

    for k, v in expected.items():
      self.assertAllClose(v, actual[k])
    self.assertEqual(expected["nodes"].shape[0], actual["n_node"])
    self.assertEqual(expected["edges"].shape[0], actual["n_edge"])

  @parameterized.named_parameters(("int_slice", False),
                                  ("tensor_slice", True))
  def test_getitem(self, use_tensor_slice):
    index = slice(1, 3)
    expected = self.graphs_dicts_out[index]

    if use_tensor_slice:
      index = slice(tf.constant(index.start), tf.constant(index.stop))

    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    graphs2 = utils_tf.get_graph(graphs_tuple, index)

    graphs2 = utils_tf.nest_to_numpy(graphs2)
    actual = utils_np.graphs_tuple_to_data_dicts(graphs2)

    for ex, ac in zip(expected, actual):
      for k, v in ex.items():
        self.assertAllClose(v, ac[k])
      self.assertEqual(ex["nodes"].shape[0], ac["n_node"])
      self.assertEqual(ex["edges"].shape[0], ac["n_edge"])

  @parameterized.named_parameters(
      ("index_bad_type", 1.,
       TypeError, "Index must be a valid scalar integer", False, False),
      ("index_bad_shape", [0, 1],
       TypeError, "Valid tensor indices must be scalars", True, False),
      ("index_bad_dtype", 1.,
       TypeError, "Valid tensor indices must have types", True, False),
      ("slice_bad_type_stop", 1.,
       TypeError, "Valid tensor indices must be integers", False, True),
      ("slice_bad_shape_stop", [0, 1],
       TypeError, "Valid tensor indices must be scalars", True, True),
      ("slice_bad_dtype_stop", 1.,
       TypeError, "Valid tensor indices must have types", True, True),
      ("slice_bad_type_start", slice(0., 1),
       TypeError, "Valid tensor indices must be integers", False, False),
      ("slice_with_step", slice(0, 1, 1),
       ValueError, "slices with step/stride are not supported", False, False),
  )
  def test_raises(self, index, error_type, message, use_constant, use_slice):
    if use_constant:
      index = tf.constant(index)
    if use_slice:
      index = slice(index)
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    with self.assertRaisesRegex(error_type, message):
      utils_tf.get_graph(graphs_tuple, index)


class TestNumGraphs(test_utils.GraphsTest):
  """Tests for the `get_num_graphs` function."""

  def setUp(self):
    super(TestNumGraphs, self).setUp()
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(self.graphs_dicts_in)
    self.empty_graph = graphs_tuple.map(lambda _: None,
                                        graphs.GRAPH_DATA_FIELDS)

  def test_num_graphs(self):
    graph = self.empty_graph.replace(n_node=tf.zeros([3], dtype=tf.int32))
    self.assertEqual(3, utils_tf.get_num_graphs(graph))


class TestNestToNumpy(test_utils.GraphsTest):
  """Test that graph with tf.Tensor fields get converted to numpy."""

  def setUp(self):
    super(TestNestToNumpy, self).setUp()
    self._graph = utils_tf.data_dicts_to_graphs_tuple([{
        "senders": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "receivers": tf.random.uniform([10], maxval=10, dtype=tf.int32),
        "nodes": tf.random.uniform([5, 7]),
        "edges": tf.random.uniform([10, 6]),
        "globals": tf.random.uniform([1, 8])
    }])

  def test_single_graph(self):
    numpy_graph = utils_tf.nest_to_numpy(self._graph)
    for field in graphs.ALL_FIELDS:
      self.assertIsInstance(getattr(numpy_graph, field), np.ndarray)
      self.assertNDArrayNear(
          getattr(self._graph, field).numpy(),
          getattr(numpy_graph, field), 1e-8)

  def test_mixed_graph_conversion(self):
    graph = self._graph.replace(nodes=None)
    graph = graph.map(lambda x: x.numpy(), ["edges"])

    converted_graph = utils_tf.nest_to_numpy(graph)
    self.assertIsNone(converted_graph.nodes)
    self.assertIsInstance(converted_graph.edges, np.ndarray)

  def test_nested_structure(self):
    regular_graph = self._graph
    graph_with_nested_fields = regular_graph.map(
        lambda x: {"a": x, "b": tf.random.uniform([4, 6])})

    nested_structure = [
        None,
        regular_graph,
        (graph_with_nested_fields,),
        tf.random.uniform([10, 6])]
    nested_structure_numpy = utils_tf.nest_to_numpy(nested_structure)

    tree.assert_same_structure(nested_structure, nested_structure_numpy)

    for tensor_or_none, array_or_none in zip(
        tree.flatten(nested_structure),
        tree.flatten(nested_structure_numpy)):
      if tensor_or_none is None:
        self.assertIsNone(array_or_none)
        continue

      self.assertIsNotNone(array_or_none)
      self.assertNDArrayNear(
          tensor_or_none.numpy(),
          array_or_none, 1e-8)


def _leading_static_shape(input_nest):
  return tree.flatten(input_nest)[0].shape.as_list()[0]


def _compile_with_tf_function(fn, graphs_tuple):

  input_signature = utils_tf.specs_from_graphs_tuple(
      graphs_tuple,
      dynamic_num_graphs=True,
      dynamic_num_nodes=True,
      dynamic_num_edges=True,)

  @functools.partial(tf.function, input_signature=[input_signature])
  def compiled_fn(graphs_tuple):
    assert _leading_static_shape(graphs_tuple.n_node) is None
    assert _leading_static_shape(graphs_tuple.senders) is None
    assert _leading_static_shape(graphs_tuple.nodes) is None
    return fn(graphs_tuple)

  return compiled_fn


class GraphsTupleSizeTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_graphs_tuple_size(self):
    data_dict = test_utils_tf1.generate_random_data_dict(
        (1,), (1,), (1,),
        num_nodes_range=(10, 15),
        num_edges_range=(20, 25))
    node_size_np = data_dict["nodes"].shape[0]
    edge_size_np = data_dict["edges"].shape[0]
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(2 * [data_dict])

    # Put it into a tf.function so the shapes are unknown statically.
    compiled_fn = _compile_with_tf_function(
        utils_tf.get_graphs_tuple_size, graphs_tuple)

    graphs_tuple_size = compiled_fn(graphs_tuple)
    node_size, edge_size, graph_size = graphs_tuple_size
    self.assertEqual(node_size.numpy(), node_size_np * 2)
    self.assertEqual(edge_size.numpy(), edge_size_np * 2)
    self.assertEqual(graph_size.numpy(), 2)


class MaskTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_mask(self):
    mask = utils_tf.get_mask(10, 12)
    self.assertAllClose(mask, np.concatenate((np.ones(10), np.zeros(2))))
    # If the padding is smaller than the mask, get all size of padding.
    mask = utils_tf.get_mask(10, 8)
    self.assertAllClose(mask, np.ones(8, dtype=bool))
    mask = utils_tf.get_mask(tf.constant(10), 12)
    self.assertAllClose(mask, np.concatenate((np.ones(10, dtype=bool),
                                              np.zeros(2, dtype=bool))))
    mask = utils_tf.get_mask(tf.constant(10), tf.constant(12))
    self.assertAllClose(mask, np.concatenate((np.ones(10, dtype=bool),
                                              np.zeros(2, dtype=bool))))


class PaddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("standard", False, False),
      ("standard_nested_features", False, True),
      ("experimental_unconnected_padding_edges", True, False),
  )
  def test_add_remove_padding(
      self, experimental_unconnected_padding_edges, nested_features):
    data_dict = test_utils_tf1.generate_random_data_dict(
        (7,), (8,), (9,),
        num_nodes_range=(10, 15),
        num_edges_range=(20, 25))
    node_size_np = data_dict["nodes"].shape[0]
    edge_size_np = data_dict["edges"].shape[0]
    unpadded_batch_size = 2
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(
        unpadded_batch_size * [data_dict])

    if nested_features:
      graphs_tuple = graphs_tuple.replace(
          edges=[graphs_tuple.edges, {}],
          nodes=({"tensor": graphs_tuple.nodes},),
          globals=([], graphs_tuple.globals,))

    num_padding_nodes = 3
    num_padding_edges = 4
    num_padding_graphs = 5
    pad_nodes_to = unpadded_batch_size * node_size_np + num_padding_nodes
    pad_edges_to = unpadded_batch_size * edge_size_np + num_padding_edges
    pad_graphs_to = unpadded_batch_size + num_padding_graphs

    def _get_padded_and_recovered_graphs_tuple(graphs_tuple):
      padded_graphs_tuple = utils_tf.pad_graphs_tuple(
          graphs_tuple,
          pad_nodes_to,
          pad_edges_to,
          pad_graphs_to,
          experimental_unconnected_padding_edges)

      # Check that we have statically defined shapes after padding.
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.nodes), pad_nodes_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.edges), pad_edges_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.senders), pad_edges_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.receivers), pad_edges_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.globals), pad_graphs_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.n_node), pad_graphs_to)
      self.assertEqual(
          _leading_static_shape(padded_graphs_tuple.n_edge), pad_graphs_to)

      # Check that we can remove the padding.
      graphs_tuple_size = utils_tf.get_graphs_tuple_size(graphs_tuple)
      recovered_graphs_tuple = utils_tf.remove_graphs_tuple_padding(
          padded_graphs_tuple, graphs_tuple_size)

      return padded_graphs_tuple, recovered_graphs_tuple

    # Put it into a tf.function so the shapes are unknown statically.
    compiled_fn = _compile_with_tf_function(
        _get_padded_and_recovered_graphs_tuple, graphs_tuple)
    padded_graphs_tuple, recovered_graphs_tuple = compiled_fn(graphs_tuple)

    if nested_features:
      # Check that the whole structure of the outputs are the same.
      tree.assert_same_structure(padded_graphs_tuple, graphs_tuple)
      tree.assert_same_structure(recovered_graphs_tuple, graphs_tuple)

      # Undo the nesting for the rest of the test.
      def remove_nesting(this_graphs_tuple):
        return this_graphs_tuple.replace(
            edges=this_graphs_tuple.edges[0],
            nodes=this_graphs_tuple.nodes[0]["tensor"],
            globals=this_graphs_tuple.globals[1])
      graphs_tuple = remove_nesting(graphs_tuple)
      padded_graphs_tuple = remove_nesting(padded_graphs_tuple)
      recovered_graphs_tuple = remove_nesting(recovered_graphs_tuple)

    # Inspect the padded_graphs_tuple.
    padded_graphs_tuple_data_dicts = utils_np.graphs_tuple_to_data_dicts(
        utils_tf.nest_to_numpy(padded_graphs_tuple))
    graphs_tuple_data_dicts = utils_np.graphs_tuple_to_data_dicts(
        utils_tf.nest_to_numpy(graphs_tuple))

    self.assertLen(padded_graphs_tuple, pad_graphs_to)

    # Check that the first 2 graphs from the padded_graphs_tuple are the same.
    for example_i in range(unpadded_batch_size):
      tree.map_structure(
          self.assertAllEqual,
          graphs_tuple_data_dicts[example_i],
          padded_graphs_tuple_data_dicts[example_i])

    padding_data_dicts = padded_graphs_tuple_data_dicts[unpadded_batch_size:]
    # Check that the third graph contains all of the padding nodes and edges.

    for i, padding_data_dict in enumerate(padding_data_dicts):

      # Only the first padding graph has nodes and edges.
      num_nodes = num_padding_nodes if i == 0 else 0
      num_edges = num_padding_edges if i == 0 else 0

      self.assertAllEqual(padding_data_dict["globals"],
                          np.zeros([9], dtype=np.float32))

      self.assertEqual(padding_data_dict["n_node"], num_nodes)
      self.assertAllEqual(padding_data_dict["nodes"],
                          np.zeros([num_nodes, 7], dtype=np.float32))
      self.assertEqual(padding_data_dict["n_edge"], num_edges)
      self.assertAllEqual(padding_data_dict["edges"],
                          np.zeros([num_edges, 8], dtype=np.float32))

      if experimental_unconnected_padding_edges:
        self.assertAllEqual(padding_data_dict["receivers"],
                            np.zeros([num_edges], dtype=np.int32) + num_nodes)
        self.assertAllEqual(padding_data_dict["senders"],
                            np.zeros([num_edges], dtype=np.int32) + num_nodes)
      else:
        self.assertAllEqual(padding_data_dict["receivers"],
                            np.zeros([num_edges], dtype=np.int32))
        self.assertAllEqual(padding_data_dict["senders"],
                            np.zeros([num_edges], dtype=np.int32))

    # Check that the recovered_graphs_tuple after removing padding is identical.
    tree.map_structure(
        self.assertAllEqual,
        graphs_tuple._asdict(),
        recovered_graphs_tuple._asdict())

  @parameterized.parameters(
      (None, False),
      ("edges", False),
      ("nodes", False),
      ("graphs", False),
      (None, True),
      ("edges", True),
      ("nodes", True),
      ("graphs", True),
  )
  def test_raises_not_enough_space(
      self, field_that_hits_limit, experimental_unconnected_padding_edges):
    data_dict = test_utils_tf1.generate_random_data_dict(
        (7,), (8,), (9,),
        num_nodes_range=(10, 15),
        num_edges_range=(20, 25))
    node_size_np = data_dict["nodes"].shape[0]
    edge_size_np = data_dict["edges"].shape[0]
    unpadded_batch_size = 2
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(
        unpadded_batch_size * [data_dict])

    # Padding graph needs to have at least one graph, and at least one node,
    # but should not need extra edges.
    pad_edges_to = unpadded_batch_size * edge_size_np
    pad_nodes_to = unpadded_batch_size * node_size_np + 1
    pad_graphs_to = unpadded_batch_size + 1

    if field_that_hits_limit == "edges":
      pad_edges_to -= 1
    elif field_that_hits_limit == "nodes":
      pad_nodes_to -= 1
    elif field_that_hits_limit == "graphs":
      pad_graphs_to -= 1

    def _get_padded_graphs_tuple(graphs_tuple):
      return utils_tf.pad_graphs_tuple(
          graphs_tuple,
          pad_nodes_to,
          pad_edges_to,
          pad_graphs_to,
          experimental_unconnected_padding_edges)

    # Put it into a tf.function so the shapes are unknown statically.
    compiled_fn = _compile_with_tf_function(
        _get_padded_graphs_tuple, graphs_tuple)

    if field_that_hits_limit is None:
      # Should work if the test is not supposed to hit any limit.
      compiled_fn(graphs_tuple)
    else:
      # Should raise an error.
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          "There is not enough space to pad the GraphsTuple"):
        compiled_fn(graphs_tuple)

if __name__ == "__main__":
  tf.test.main()
