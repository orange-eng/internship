# pylint: enable=line-too-long
import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np
from paddle import metric
from paddle.vision import models
from paddle.vision.transforms import transforms
from paddle.vision.models import resnet18

import tree
import paddle
import paddle.vision.transforms as T
from paddle.metric import Accuracy

# 导入定义的数据类
from dataset import MyDataset
# from learning_to_simulate import learned_simulator
# from learning_to_simulate import noise_utils
# from learning_to_simulate import reading_utils
import learned_simulator
import noise_utils
import reading_utils

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', 'tmp/datasets/WaterRamps', help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
#flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_integer('num_steps', int(10), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', 'tmp/models/WaterRamps_10',
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', 'tmp/rollouts/WaterRamps_10',
                    help='The path for saving outputs (e.g. rollouts).')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  return paddle.equal(particle_types, KINEMATIC_PARTICLE_ID)
  # tf.equal(x, y)
  # 在矩阵或者向量x和y中，如果相同位置元素相等，
  # 返回True，否则返回False
  # 在该代码中，等于3的元素为True,不等于3的元素为False

def prepare_inputs(tensor_dict):
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = paddle.transpose(pos, perm=[1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]
  # Compute the number of particles per example.
  num_particles = paddle.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  return tensor_dict, target_position

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)




def main(_):
  transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0.5,std=0.5)
  ])
  train_img_path = []
  train_label = []
  train_dataset = MyDataset(image=train_img_path,lable=train_label,transform=transform)
  train_loader = paddle.io.DataLoader(train_dataset,places=paddle.CPUPlace(),batch_size=2,shuffle=True)
  model = resnet18(pretrained=True,num_classes=102,with_pool=True)
  model = paddle.Model(model)
  optim = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())

  """Train or evaluates the model."""
  if FLAGS.mode == 'train':
    model.prepare(
      optimizer=optim,
      loss=paddle.nn.MSELoss(),
      metric=Accuracy() # topk计算准确率的top个数，默认是1
    )
    model.fit(train_loader,
      epochs=2,
      verbose=1,
    )
    model.evaluate(train_dataset,batch_size=2,verbose=1)
    model.save('inference_model',training=False)

  elif FLAGS.mode == 'eval_rollout':
    metadata = _read_metadata(FLAGS.data_path)


if __name__ == '__main__':
  app.run(main)
