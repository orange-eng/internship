
#codig:utf-8
import tensorflow as tf
import numpy as np
c = np.random.random([5,1])  ##随机生成一个5*1的数组
b = tf.nn.embedding_lookup(c, [1, 3]) ##查找数组中的序号为1和3的
 
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))
    print(c)
