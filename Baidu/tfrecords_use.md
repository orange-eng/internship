# tfrecords_uses
- tfrecord是tensorflow用于存储数据集的一种格式。在学习deepmind的开源代码时，对tfrecord数据进行了研究，这里简单总结一下，以便于后续的学习。

## 将数据存储为tfrecord
这里先直接放上代码如下（亲测有效）：
```python
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

```
这段代码的意义在于，将data和label存入到data.tfrecords文件当中。可以观察到，save_tfrecods函数实际上是定义了一种features，传入的数据（data和label）需要按照features的格式进行存储。

## 展示tfrecords数据的内容
这里要注意，存储tfrecords是使用迭代的方式来进行，因此读取时，同样需要迭代来读取。代码如下（亲测有效）：
```python

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
```

## 将照片存储为tfrecords
代码如下（亲测有效）：
```python
import tensorflow as tf

def write_test(input, output):
    # 借助于TFRecordWriter 才能将信息写入TFRecord 文件
    writer = tf.python_io.TFRecordWriter(output)
    # 读取图片并进行解码
    image = tf.read_file(input)
    image = tf.image.decode_jpeg(image)
    with tf.Session() as sess:
        image = sess.run(image)
        shape = image.shape
        # 将图片转换成string
        image_data = image.tostring()   # 把图像数据转化为二进制数组string存储
        print("img_data=",image)
        #print("img_type=",type(image))
        #print("img_len=",len(image_data))   # 所有像素点的个数
        name = bytes('cat', encoding='utf-8')
        #print("type_name=",type(name))
        print("name=",name)
        # 创建Example对象，并将Feature一一对应填充进去
        example = tf.train.Example(features=tf.train.Features(feature={
             'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
             'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1], shape[2]])),
             'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
        }
        ))
        # 将example序列化成string 类型，然后写入。
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    input_photo = 'train_imgs/4.png'
    output_file = 'tfrecord/4.tfrecord'
    write_test(input_photo, output_file)


```
