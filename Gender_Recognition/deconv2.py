
from __future__ import print_function, division
import numpy as np
import torch

import torch
import os
import numpy as np
import torch.nn as nn

def zy_deconv(img, in_channels, out_channels, kernels,bias, stride=1, padding=0,output_padding=0):
    #得到参数
    N, C, H, W = img.shape
    kc,kc_in,kh, kw = kernels.shape
    p = padding

    #间隔填充
    if  stride>1:
       gap_img=np.zeros([N,in_channels,stride*H-1,stride*W-1])
       for n in range(N):
           for in_c in range(in_channels):
               for  x in range(H):
                   for y in range(W):
                        gap_img[n][in_c][stride*x][stride*y]=img[n][in_c][x][y]
       N, C, H, W = gap_img.shape
       img=gap_img


    #四周填充
    p=kh-p-1
    if p:
        img = np.pad(img, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')  # padding along with all axis

    out_h = (H + 2 * p - kh) + 1
    out_w = (W + 2 * p - kw) + 1

    #右边和下方填充
    if output_padding:
        temp = np.zeros([N,in_channels, H+2*p+output_padding, W+2*p+output_padding])
        #先将原有的数据填入
        for n in range(N):
             for in_c in range(in_channels):
                for  x in range(H+2*p):
                   for y in range(W+2*p):
                     temp[n][in_c][x][y] =img[n][in_c][x][y]

                for  a  in range(output_padding):
                    #对下方与右方填入output_padding行（列）的0
                    for  i in range (W+2*p) :
                        temp[n][in_c][H+2*p+a][i]=0
                    for  j  in range (H+2*p+output_padding) :
                        temp[n][in_c][j][W+2*p+a]=0
        img=temp
        out_h=  out_h+output_padding
        out_w=  out_w+output_padding

    #卷积
    outputs = np.zeros([N, out_channels, out_h, out_w])
    for n in range(N):
        for out_c in range(out_channels):
            for in_c in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                             for x in range(kh):
                                 for y in range(kw):
                                     #这里的矩阵旋转了180度
                                    outputs[n][out_c][h][w] += img[n][in_c][h + x][w + y] * kernels[in_c][out_c][kh-x-1][kw-y-1]
            #添加偏置
            if in_c == in_channels - 1:
                            outputs[n][out_c][:][:] += bias[out_c]

    #转化为tensor格式，内部数据为float32的形式
    outputs = torch.tensor(outputs, dtype=torch.float32)
    return outputs

img = np.asarray(
        [[
            [
                [1, 2, 3],
                [6, 5, 4],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [6, 5, 4],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [6, 5, 4],
                [7, 8, 9]
            ],
        ]]
)

#将img转化为tensor类型
img= torch.tensor(img, dtype=torch.float32)
deconv= nn.ConvTranspose2d(3, 2, 3, 2, 1,output_padding=1)#输入通道，输出通道，卷积核大小，步长，padding,output_padding

#进行反卷积操作
out=deconv(img)
print("卷积核：")
print(deconv.weight)
print("偏置：")
print(deconv.bias)
print("pytorch自带反卷积函数运行结果：")
print(out)

#提取出卷积核与偏置矩阵
kernels=deconv.weight
kernals=kernels.detach().numpy()
bias=deconv.bias
bias=bias.detach().numpy()

in_channels = 3
out_channels = 2

#将提取出来的卷积核与偏置矩阵代入自编的卷积函数中，对比自编函数与pytorch自带函数的结果
outputs = zy_deconv(img, in_channels, out_channels, kernels,bias, stride=2, padding=1, output_padding=1)
print("自编反卷积函数运行结果：")
print(outputs)
