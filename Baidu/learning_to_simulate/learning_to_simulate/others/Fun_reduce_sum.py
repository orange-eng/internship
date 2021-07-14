

##############################################################

import tensorflow as tf
 
x = [[1,2,3],
     [1,2,3]]
 
xx = tf.cast(x,tf.float32)
 
sum_all = tf.reduce_sum(xx, keepdims=False)
sum_0 = tf.reduce_sum(xx, axis=0, keepdims=False)
sum_1 = tf.reduce_sum(xx, axis=1, keepdims=False)
 
 
with tf.Session() as sess:
    s_a,s_0,s_1 = sess.run([sum_all, sum_0, sum_1])
 
print(s_a)
print(s_0)
print(s_1)
 
 
###  输出结果为 ###
# 12.0
# [ 2.  4.  6.]
# [ 6.  6.]