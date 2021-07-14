
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

def batch_concat(dataset, batch_size):
  windowed_ds = dataset.window(batch_size)
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)
  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))

def prepare_inputs(tensor_dict):
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])
  # perm:控制转置的操作,以perm = [0,1,2] 3个维度的数组为例, 0–代表的是最外层的一维, 
  # 1–代表外向内数第二维, 2–代表最内层的一维,这种perm是默认的值.
  # 如果换成[1,0,2],就是把最外层的两维进行转置，比如原来是2乘3乘4，
  # 经过[1,0,2]的转置维度将会变成3乘2乘4
  target_position = pos[:, -1]
  tensor_dict['position'] = pos[:, :-1]
  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
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
    split_with_window = functools.partial(
        reading_utils.split_trajectory,
        window_length=INPUT_SEQUENCE_LENGTH + 1)
    ds = ds.flat_map(split_with_window)
    ds = ds.map(prepare_inputs)
    ds = batch_concat(ds, batch_size)
    return ds
  return input_fn

input_fn=get_input_fn('tmp/datasets/WaterRamps', 1,
                    mode='one_step_train', split='test')
# 实例化了一个Iterator
iterator = input_fn().make_one_shot_iterator()  # 单次迭代
# 从iterator里取出一个元素
one_element = iterator.get_next()               # 获得数据
with tf.Session() as sess:
    for i in range(1):
        result = sess.run(one_element)
        print("result_0=",result[0])
        print("result_1=",result[1])
        #print("len_result_1=",len(result[1]))
        particle_type = result[0]['particle_type']
        position = result[0]['position']
        print("position=\n",position)
        print("===========================================================")
