

###################################################
import tensorflow as tf
a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
condition1 = [[True,False,False],
             [False,True,True]]
condition2 = [[True,False,False],
             [False,True,False]]
with tf.Session() as sess:
    print(sess.run(tf.where(condition1)))
    print(sess.run(tf.where(condition2)))


x = [[1,2,3],[4,5,6]]
y = [[7,8,9],[10,11,12]]
condition3 = [[True,False,False],
             [False,True,True]]
condition4 = [[True,False,False],
             [True,True,False]]
with tf.Session() as sess:
    print(sess.run(tf.where(condition3,x,y)))
    print(sess.run(tf.where(condition4,x,y)))  

global_step = tf.train.get_global_step()
print(global_step)


###################################################################################
