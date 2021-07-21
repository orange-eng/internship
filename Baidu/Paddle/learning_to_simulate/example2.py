import numpy as np

import paddle
import paddle.nn as nn

import paddle
import numpy as np
import os
import json

# in1 = np.array([[1, 2, 3],
#                 [4, 5, 6]])
# in2 = np.array([[11, 12, 13, 20],
#                 [14, 15, 16, 21]])

# x1 = paddle.to_tensor(in1)
# x2 = paddle.to_tensor(in2)
# out1 = paddle.concat(x=[x1, x2], axis=-1)
# print(out1)

# in1 = np.array([[0.1, 0.9],
#                 [0.1, 0.9]])
# print(in1[:,0])

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())


metadata = _read_metadata('')
print(metadata)
print(metadata['bounds'])
print(metadata['sequence_length'])