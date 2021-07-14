

import tensorflow as tf
 
with tf.Session() as sess:
    x = tf.constant([1.8, 2.2], dtype=tf.float32)
    print(x)
    b = tf.dtypes.cast(x, tf.int32)
    print(b)
 
# 输出结果：
# Tensor("Const:0", shape=(2,), dtype=float32)
# Tensor("Cast:0", shape=(2,), dtype=int32)
