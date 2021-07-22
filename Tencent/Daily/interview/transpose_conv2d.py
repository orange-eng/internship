input_data=[
               [[1,0,1],
                [0,2,1],
                [1,1,0]],
 
               [[2,0,2],
                [0,1,0],
                [1,0,0]],
 
               [[1,1,1],
                [2,2,0],
                [1,1,1]],
 
               [[1,1,2],
                [1,0,1],
                [0,2,2]]
 
            ]
weights_data=[ 
              [[[ 1, 0, 1],
                [-1, 1, 0],
                [ 0,-1, 0]],
               [[-1, 0, 1],
                [ 0, 0, 1],
                [ 1, 1, 1]],
               [[ 0, 1, 1],
                [ 2, 0, 1],
                [ 1, 2, 1]], 
               [[ 1, 1, 1],
                [ 0, 2, 1],
                [ 1, 0, 1]]],
 
              [[[ 1, 0, 2],
                [-2, 1, 1],
                [ 1,-1, 0]],
               [[-1, 0, 1],
                [-1, 2, 1],
                [ 1, 1, 1]],
               [[ 0, 0, 0],
                [ 2, 2, 1],
                [ 1,-1, 1]], 
               [[ 2, 1, 1],
                [ 0,-1, 1],
                [ 1, 1, 1]]]  
           ]

print(input_data)

import numpy as np
def compute_conv(fm,kernel,stride=1):
    [h,w] = fm.shape
    [k,_] = kernel.shape
    r = int(k/2)
    # padding
    padding_fm = np.zeros([h+2,w+2])
    # 保存计算结果
    res  =np.zeros([h,w])
    for i in range(0,h):
        for j in range(0,w):
            roi = padding_fm[i:i+r,j:j+r]