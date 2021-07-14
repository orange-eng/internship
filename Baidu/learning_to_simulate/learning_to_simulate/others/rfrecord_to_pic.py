
# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import learned_simulator
import noise_utils
import reading_utils

import collections
import functools

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
#############################################
## 读取一张图片的数据
#############################################

def _parse_record(example_photo):
    features = {
        'name': tf.FixedLenFeature((), tf.string),
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_photo,features=features)
    return parsed_features

#############################################
## 读取WaterRamps数据
#############################################
def water_ramps_parse_record(example_photo):
    features = {
        'key': tf.FixedLenFeature((),tf.int64),
        'particle_type': tf.FixedLenFeature((),tf.string)
    }
    parsed_features = tf.parse_single_example(example_photo,features=features)
    return parsed_features

def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.
  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.
  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.
  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.

  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).
  Returns:
    A tuple of input features and target positions.
  """
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



def read_test(input_file):
    # 用dataset读取TFRecords文件
    dataset = tf.data.TFRecordDataset(input_file)
    print("dataset=",dataset)
    #dataset = dataset.map(_parse_record)
    dataset = dataset.map(water_ramps_parse_record)
    print("dataset_2=",dataset)

    split_with_window = functools.partial(
        reading_utils.split_trajectory,
        window_length=INPUT_SEQUENCE_LENGTH + 1)
    ds = dataset.flat_map(split_with_window)
    print("ds=",ds)
    # Splits a chunk into input steps and target steps
    ds = ds.map(prepare_inputs)

    iterator = dataset.make_one_shot_iterator()
'''
    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        # name = features['name']
        # name = name.decode()
        # img_data = features['data']
        key = features['key']
        particle_type = features['particle_type']
        print("key=",key)
        # print("img_data=",img_data)
        # print("type_img_data=",type(img_data))
        # shape = features['shape']
        # print("shape=",shape)
        # print("==============")
        # print(type(shape))
        # print(len(img_data))

        # 从bytes数组中加载图片原始数据，并重新reshape，它的结果是 ndarray 数组
        # img_data = np.fromstring(img_data, dtype=np.uint8)
        particle_type = np.fromstring(particle_type,dtype=np.uint8)
        #print("particle=\n",particle_type)
        print("shape_particle=",len(particle_type))    # shape = 11552

        # image_data = np.reshape(img_data, shape)
        # plt.figure()
        # # 显示图片
        # plt.imshow(image_data)
        # plt.show()

        # 将数据重新编码成jpg图片并保存
        # img = tf.image.encode_jpeg(image_data)
        # tf.gfile.GFile('cat_encode.jpg', 'wb').write(img.eval())
'''
if __name__ == '__main__':
    read_test("tmp/datasets/WaterRamps/train.tfrecord")
