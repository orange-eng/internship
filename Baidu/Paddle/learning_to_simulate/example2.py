import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
 
import paddle
import numpy as np

in1 = np.array([[1, 2, 3],
                [4, 5, 6]])
in2 = np.array([[11, 12, 13, 20],
                [14, 15, 16, 21]])

x1 = paddle.to_tensor(in1)
x2 = paddle.to_tensor(in2)
out1 = paddle.concat(x=[x1, x2], axis=-1)
# out2 = paddle.concat(x=[x1, x2], axis=0)
# out3 = paddle.concat(x=[x1, x2], axis=zero)
print(out1)