import tensorflow as tf
import numpy as np
 
# 输入：1张图片，尺寸28*28 高宽，通道数3
x = np.ones((1, 28, 28, 3), dtype=np.float32)
# 卷积核尺寸4x4 ，5表输出通道数，3代表输入通道数
w = np.ones((4, 4, 5, 3), dtype=np.float32)
#扩大2倍
output = tf.nn.conv2d_transpose(x, w, (1, 56, 56, 5), [1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    m = sess.run(output)
    print(m.shape)
