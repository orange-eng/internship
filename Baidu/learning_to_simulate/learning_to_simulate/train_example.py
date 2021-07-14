
import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np

import tensorflow.compat.v1 as tf
import tree

import learned_simulator
import noise_utils
import reading_utils

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)
def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)


def batch_concat(dataset, batch_size):
  """We implement batching as concatenating on the leading axis."""

  # We create a dataset of datasets of length batch_size.
  windowed_ds = dataset.window(batch_size)

  # The plan is then to reduce every nested dataset by concatenating. We can
  # do this using tf.data.Dataset.reduce. This requires an initial state, and
  # then incrementally reduces by running through the dataset

  # Get initial state. In this case this will be empty tensors of the
  # correct shape.
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)

  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))



def _get_simulator(model_kwargs, metadata, acc_noise_std, vel_noise_std):
  """Instantiates the simulator."""
  # Cast statistics to numpy so they are arrays when entering the model.
  cast = lambda v: np.array(v, dtype=np.float32)    # cast 把自变量变为numpy类型
  acceleration_stats = Stats(
      cast(metadata['acc_mean']),
      _combine_std(cast(metadata['acc_std']), acc_noise_std))
  velocity_stats = Stats(
      cast(metadata['vel_mean']),
      _combine_std(cast(metadata['vel_std']), vel_noise_std))
  normalization_stats = {'acceleration': acceleration_stats,
                         'velocity': velocity_stats}
  if 'context_mean' in metadata:
    context_stats = Stats(
        cast(metadata['context_mean']), cast(metadata['context_std']))
    normalization_stats['context'] = context_stats

  simulator = learned_simulator.LearnedSimulator(
      num_dimensions=metadata['dim'],
      connectivity_radius=metadata['default_connectivity_radius'],
      graph_network_kwargs=model_kwargs,
      boundaries=metadata['bounds'],
      num_particle_types=NUM_PARTICLE_TYPES,
      normalization_stats=normalization_stats,
      particle_type_embedding_size=16)
  return simulator


def get_one_step_estimator_fn(data_path,
                              noise_std,
                              latent_size=128,
                              hidden_size=128,
                              hidden_layers=2,
                              message_passing_steps=10):
  """Gets one step model for training simulation."""
  metadata = _read_metadata(data_path)

  model_kwargs = dict(
      latent_size=latent_size,              # 128
      mlp_hidden_size=hidden_size,          # 128
      mlp_num_hidden_layers=hidden_layers,  # 2
      num_message_passing_steps=message_passing_steps)  # 10

  def estimator_fn(features, labels, mode):
    target_next_position = labels
    simulator = _get_simulator(model_kwargs, metadata,
                               vel_noise_std=noise_std,
                               acc_noise_std=noise_std)
    print("feeature=",features['position'])


    # Sample the noise to add to the inputs to the model during training.
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        features['position'], noise_std_last_step=noise_std)
    non_kinematic_mask = tf.logical_not(
        get_kinematic_mask(features['particle_type']))
    noise_mask = tf.cast(                                                     # tf.cast函数用于转换数据类型
        non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]   # tf.newaxis 给tensor增加维度
    sampled_noise *= noise_mask     # sampled_noise * noise_mask
    # sampled_noise就是需要的噪声
    # Get the predictions and target accelerations.
    pred_target = simulator.get_predicted_and_target_normalized_accelerations(
        next_position=target_next_position,
        position_sequence=features['position'],
        position_sequence_noise=sampled_noise,
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=features.get('step_context'))
    pred_acceleration, target_acceleration = pred_target
    # Calculate the loss and mask out loss on kinematic particles/
    loss = (pred_acceleration - target_acceleration)**2
    num_non_kinematic = tf.reduce_sum(      # tf.reduce_sum用于计算张量tensor沿着某一维度的和，可以在求和后降维
        tf.cast(non_kinematic_mask, tf.float32))
    loss = tf.where(non_kinematic_mask, loss, tf.zeros_like(loss))
    # condition， x, y 相同维度，condition是bool型值，True/False
    # 返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
    # x只负责对应替换True的元素，y只负责对应替换False的元素，x，y各有分工
    # non_kinematic_mask为false，那么置零，否则置为loss值
    loss = tf.reduce_sum(loss) / tf.reduce_sum(num_non_kinematic)
    global_step = tf.train.get_global_step()
    # global_step是比较常用的变量，如果和普通Variable混在一起就降低了代码的可读性。
    # 返回的是global_step作为name的tensor
    # Set learning rate to decay from 1e-4 to 1e-6 exponentially.
    min_lr = 1e-6   # 学习率
    lr = tf.train.exponential_decay(learning_rate=1e-4 - min_lr,  # 初始学习率
                                    global_step=global_step,      # 当前迭代次数
                                    decay_steps=int(5e6),         # 衰减速度（在迭代到该次数时，学习率衰减为learning_rate * decay_rate）
                                                                  # 相当于，迭代到5e6次时，缩小为原来的0.1
                                    decay_rate=0.1) + min_lr      # decay_rate 学习率衰减系数，在0~1之间
    # 1.首先使用较大学习率(目的：为快速得到一个比较优的解);
    # 2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss, global_step)
    # Calculate next position and add some additional eval metrics (only eval).
    predicted_next_position = simulator(
        position_sequence=features['position'],
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=features.get('step_context'))
    predictions = {'predicted_next_position': predicted_next_position}
    eval_metrics_ops = {
        'loss_mse': tf.metrics.mean_squared_error(
            pred_acceleration, target_acceleration),
        'one_step_position_mse': tf.metrics.mean_squared_error(
            predicted_next_position, target_next_position)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions=predictions,
        eval_metric_ops=eval_metrics_ops)
  return estimator_fn


def prepare_inputs(tensor_dict):
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]
  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position

#########################################
### get_input
#########################################
def get_input_fn(data_path, batch_size, mode, split):
  def input_fn():
    metadata = _read_metadata(data_path)
    # Create a tf.data.Dataset from the TFRecord.
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if mode.startswith('one_step'):
      # Splits an entire trajectory into chunks of 7 steps.
      # Previous 5 velocities, current velocity and target.
      split_with_window = functools.partial(
          reading_utils.split_trajectory,
          window_length=INPUT_SEQUENCE_LENGTH + 1)
      ds = ds.flat_map(split_with_window)
      # Splits a chunk into input steps and target steps
      ds = ds.map(prepare_inputs)
      ds = batch_concat(ds, batch_size)
    else:
      raise ValueError(f'mode: {mode} not recognized')
    return ds

  return input_fn

input_fn=get_input_fn('tmp/datasets/WaterRamps', 1,
                    mode='one_step_train', split='test')
# input_fn就是输入的数据

# 实例化了一个Iterator
iterator = input_fn().make_one_shot_iterator()  # 单次迭代
# 从iterator里取出一个元素
one_element = iterator.get_next()               # 获得数据
with tf.Session() as sess:
    num = 0
    for i in range(3600):
        result = sess.run(one_element)
        if i % 600 == 0:
            particle_type = result[0]['particle_type']
            position = result[0]['position']
            # if particle_type[0] == 5:   #出现障碍物部分时
            num += 1
            print("num=",num)     # 统计全部为水滴的画面个数
            #print("particle_first=",particle_type[0])
            #print("position_first=",position[0])
            print("particle=",particle_type)
            key = np.unique(particle_type)
            result = {}
            for k in key:
                mask = (particle_type == k)
                y_new = particle_type[mask]
                v = y_new.size
                result[k] = v
            print(result)
            #print("len_particle=",len(particle_type))
            # print("position_first=",position)
            # print("len_position=",len(position))
            print("===========================================================")
