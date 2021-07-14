import paddle
from paddle import fluid

# add to input variable to program to accept inputs
images  = fluid.layers.data(name='pixel',shape=[1,28,28],dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

# construct the model with input variable
conv_pool_1 = fluid.nets.simple_img_conv_pool(
    input = images,filter_size = 5,
    num_filters = 20, pool_size = 2,
    pool_stride = 2, act = 'relu')

conv_pool_2 = fluid.nets.simple_img_conv_pool(
    input = conv_pool_1,filter_size = 5,
    num_filters = 50, pool_size = 2,
    pool_stride = 2, act = 'relu')

SIZE = 10
input_shape = conv_pool_2.shape
#param_shape = [reduce(lambda a,b:a*b,input_shape[1:],1)] + [SIZE]
