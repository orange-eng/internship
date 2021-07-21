

# from learning_to_simulate import connectivity_utils
# from learning_to_simulate import graph_network
import connectivity_utils_paddle
import graph_network_paddle
import paddle.nn as nn

STD_EPSILON = 1e-8


class LearnedSimulator(object):
  def __init__(
      self,
      num_dimensions,
      connectivity_radius,
      graph_network_kwargs,
      boundaries,
      normalization_stats,
      num_particle_types,
      particle_type_embedding_size,
      name="LearnedSimulator"):

    super().__init__(name=name)
    self._connectivity_radius = connectivity_radius # 0.015
    self._num_particle_types = num_particle_types   # 9
    self._boundaries = boundaries                   # [[0.1,0.9],[0.1,0.9]]
    self._normalization_stats = normalization_stats 

    with self._enter_variable_scope():
      self._graph_network = graph_network_paddle.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

  def _build(self, position_sequence, n_particles_per_example):
    input_graphs_tuple = self._encoder_preprocessor(
        position_sequence, n_particles_per_example)

    normalized_acceleration = self._graph_network(input_graphs_tuple)

    next_position = self._decoder_postprocessor(
        normalized_acceleration, position_sequence)

    return next_position

  def _encoder_preprocessor(
      self, position_sequence, n_node):
    # Extract important features from the position_sequence.
    most_recent_position = position_sequence[:, -1]
    velocity_sequence = time_diff(position_sequence)  # Finite-difference.

    # Get connectivity of the graph.
    (senders, receivers, n_edge
     ) = connectivity_utils_paddle.compute_connectivity_for_batch_pyfunc(
         most_recent_position, n_node, self._connectivity_radius)

    # Collect node features.
    node_features = []
    # node_feat总共包含以下几项：
    # 1.flat_velocity_Sequence
    # 2.distance_to_lower_boundary
    # 3.distance_to_upper_boundary
    # 4.particle_type_embeddings

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats.mean) / velocity_stats.std

    flat_velocity_sequence = snt.MergeDims(start=1, size=2)(
        normalized_velocity_sequence)
  #   node_features.append(flat_velocity_sequence)

  #   # Normalized clipped distances to lower and upper boundaries.
  #   # boundaries are an array of shape [num_dimensions, 2], where the second
  #   # axis, provides the lower/upper boundaries.
  #   boundaries = tf.constant(self._boundaries, dtype=tf.float32)
  #   distance_to_lower_boundary = (
  #       most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
  #   distance_to_upper_boundary = (
  #       tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
  #   distance_to_boundaries = tf.concat(   # 拼接张量操作
  #       [distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
  #   normalized_clipped_distance_to_boundaries = tf.clip_by_value(   # tf.clip_by_value(V, min, max), 截取V使之在min和max之间
  #       distance_to_boundaries / self._connectivity_radius, -1., 1.)
  #   # 将距离控制在-1~1之间
  #   node_features.append(normalized_clipped_distance_to_boundaries)

  # #   # Particle type.
  #   # tf.nn.embedding_lookup查找数组中的序号为particle_types的元素
  #   if self._num_particle_types > 1:
  #     particle_type_embeddings = tf.nn.embedding_lookup(
  #         self._particle_type_embedding, particle_types)
  #     node_features.append(particle_type_embeddings)

  # #   # Collect edge features.
  #   edge_features = []
  #   #1. normalized_realative_displacements = sender - receiver / radius
  #   #2. normalized_relative_distances       向量/矩阵的范数
  #   # Relative displacement and distances normalized to radius
  #   normalized_relative_displacements = (
  #       tf.gather(most_recent_position, senders) -
  #       tf.gather(most_recent_position, receivers)) / self._connectivity_radius
  #   # sender - receiver / radius
  #   edge_features.append(normalized_relative_displacements)

  #   normalized_relative_distances = tf.norm(
  #       normalized_relative_displacements, axis=-1, keepdims=True)
  #   edge_features.append(normalized_relative_distances)

  #   # Normalize the global context.
  #   if global_context is not None:
  #     context_stats = self._normalization_stats["context"]
  #     # Context in some datasets are all zero, so add an epsilon for numerical
  #     # stability.
  #     global_context = (global_context - context_stats.mean) / tf.math.maximum(
  #         context_stats.std, STD_EPSILON)

  #   return gn.graphs.GraphsTuple(
  #       nodes=tf.concat(node_features, axis=-1),
  #       edges=tf.concat(edge_features, axis=-1),
  #       globals=global_context,  # self._graph_net will appending this to nodes.
  #       n_node=n_node,
  #       n_edge=n_edge,
  #       senders=senders,
  #       receivers=receivers,
  #       )

  def _decoder_postprocessor(self, normalized_acceleration, position_sequence):

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats.std
        ) + acceleration_stats.mean

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def get_predicted_and_target_normalized_accelerations(
      self, next_position, position_sequence_noise, position_sequence,
      n_particles_per_example, global_context=None, particle_types=None):  # pylint: disable=g-doc-args
    """Produces normalized and predicted acceleration targets.

    Args:
      next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
        with the positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence, n_node, global_context, particle_types: Inputs to the
        model as defined by `_build`.

    Returns:
      Tensors of shape [num_particles_in_batch, num_dimensions] with the
        predicted and target normalized accelerations.
    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    input_graphs_tuple = self._encoder_preprocessor(
        noisy_position_sequence, n_particles_per_example, global_context,
        particle_types)
    predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_position + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(self, next_position, position_sequence):
    """Inverse of `_decoder_postprocessor`."""

    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats.mean) / acceleration_stats.std
    return normalized_acceleration


# time_diff可以得到美意时刻的差值
def time_diff(input_sequence):
  return input_sequence[:, 1:] - input_sequence[:, :-1]

