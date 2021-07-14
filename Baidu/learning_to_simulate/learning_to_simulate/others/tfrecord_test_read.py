
from typing import Callable
import collections
import functools
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tree
####################################
### Function
####################################
def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())
##########################################################
### 展示test.tfrecords里面的内容
##########################################################
import tensorflow as tf
def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.int64),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['data'],parsed_features['label']


def test_parse_function(example_proto):
    features = {"key": tf.FixedLenFeature((), tf.int64),
                "particle_type": tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['key'],parsed_features['particle_type']

def load_tfrecords(srcfile):
    output_key = []
    output_particle_type = []
    sess = tf.Session()
    dataset = tf.data.TFRecordDataset(srcfile)
    dataset = dataset.map(test_parse_function)
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    # while True:
    #     try:
    #         key, particle_type = sess.run(next_data)
    #         print("key=",key)
    #         output_key.append(key)
    #         output_particle_type.append(particle_type)
            
    #     except tf.errors.OutOfRangeError:
    #         break
    key, particle_type = sess.run(next_data)
    print("key=",key)
    output_key.append(key)
    output_particle_type.append(particle_type)

    return output_key,output_particle_type

# output_data,output_label = load_tfrecords(srcfile='./tmp/datasets/WaterRamps/test.tfrecord')
# print(output_data)
# print(output_label)


# #data_path = 'learning_to_simulate/others/tfrecord'

# metadata = _read_metadata(data_path)
# #确认tfrecord的内容
# ex=next(tf.python_io.tf_record_iterator(data_path + '/train.tfrecord'))
# print(tf.train.Example.FromString(ex))
# print("==========================================")
# print(tf.train.Example.FromString(ex))


