'''
#获取.ckpt文件里面的数据
import os

import numpy as np
from tensorflow.python import pywrap_tensorflow
checkpoint_path='/tmp/models/WaterRamps_1000000/model.ckpt-1000000'

print(checkpoint_path)
#read data from checkpoint file
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
 
 
data_print=np.array([])
for key in var_to_shape_map:
    print('tensor_name',key)
    ckpt_data=np.array(reader.get_tensor(key))#cast list to np arrary
    ckpt_data=ckpt_data.flatten()#flatten list
    data_print=np.append(data_print,ckpt_data,axis=0)
 
print(data_print,data_print.shape,np.max(data_print),np.min(data_print),np.mean(data_print))
'''

import tensorflow as tf
g = tf.Graph() 
with g.as_default() as g: 
    tf.train.import_meta_graph('./tmp/models/WaterRamps_1000000/model.ckpt-1000000.meta') 
 
with tf.Session(graph=g) as sess: 
    file_writer = tf.summary.FileWriter(logdir='./tmp/your_out_file', graph=g)
