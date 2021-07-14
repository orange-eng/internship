import os
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":tf.train.Feature(int64_list = tf.train.Int64List(value = [data[i]])),
                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)

data = [0,1,2,3,4,5,6,7,8,9]
label = [0,0,0,0,0,1,1,1,1,1]
# 存储好tfrecord数
print("len=",data)
print("label=",label)
save_tfrecords(data, label, "./img_sample/data.tfrecords")

##########################################################
### 展示tfrecords里面的内容
##########################################################

from absl import app
from absl import flags
from absl import logging
import numpy as np

def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.int64),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['data'],parsed_features['label']

def load_tfrecords(srcfile):
    output_data = []
    output_label = []
    sess = tf.Session()
    dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
    dataset = dataset.map(_parse_function) # parse data into tensor
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    while True:
        try:
            data, label = sess.run(next_data)
            print("data=",data)
            output_data.append(data)
            output_label.append(label)
            #print("label=",label)
        except tf.errors.OutOfRangeError:
            break
    return output_data,output_label

output_data,output_label = load_tfrecords(srcfile='./img_sample/data.tfrecords')
print(output_data)
print(output_label)