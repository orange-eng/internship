
import functools
import numpy as np
from numpy.lib.function_base import append
import paddle
import pandas as pd

def str_to_float(x):
    return float(x)

def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = paddle.to_tensor(np.array(out))
  return out

# file = pd.read_csv('dataset/four_balls/csv/0.csv')

# df = pd.DataFrame(file)
# # for i in range(1):
# #   document = df[i:i+1]
# #   print(document,'\n')
# print(df[0:1])

import csv
rows = []
with open('dataset/four_balls/csv/0.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      float_row = list(map(str_to_float,row))
      rows.append(float_row)

# 存取每个点的position_sequence
node_1_position = []
node_2_position = []
node_3_position = []
node_4_position = []
print(rows)
print("="*80)
for i in range(len(rows)):
  node_1_position.append([rows[i][0],rows[i][1]])
  node_2_position.append([rows[i][2],rows[i][3]])
  node_3_position.append([rows[i][4],rows[i][5]])
  node_4_position.append([rows[i][6],rows[i][7]])


for k in range(len(node_1_position)):
  print(node_1_position[k])