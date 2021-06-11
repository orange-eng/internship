


import numpy as np
from numpy.lib.arraypad import pad
 
 
f = np.array([[1,2,1],
[1,0,0],
[-1,0,1]])

img = np.array([
       [2,3,7,4,6,2,9],
       [6,6,9,8,7,4,3],
       [3,4,8,3,8,9,7],
       [7,8,3,6,6,3,4],
       [4,2,1,8,3,4,6],
       [3,2,4,1,9,8,3],
       [0,1,3,9,2,1,4]])

#######################################
## 卷积操作
#######################################
def conv2(img,f,stride):
    img_w,img_h = img.shape
    w,h = f.shape
    out_w = (img_w - w)//stride + 1
    out_h = (img_h - h)//stride + 1

    arr = np.zeros(shape=(out_w,out_h))
    for g in range(out_h):
        for t in range(out_w):
            s = 0
            # 卷积相乘
            for i in range(w):
                for j in range(h):
                    s = s + img[i+g*stride][j+t*stride]*f[i][j]
            arr[g][t] = s
    
    return arr

#######################################
## 卷积操作
#######################################

def numpy_conv(input,kernel,stride,padding=0):
    H,W = input.shape
    
    kernel_size = kernel.shape[1]
    out_w = (W + 2*padding - kernel_size)//stride + 1
    out_h = (H + 2*padding - kernel_size)//stride + 1
    out_channel = kernel.shape[0]
    print("out_w,out_h=",out_w,out_h,out_channel)
    result = np.zeros(shape=(out_channel,out_w,out_h))

    for r in range(0,H - kernel_size + 1):
        for c in range(0,W - kernel_size +1):
            for i in range(0,out_channel):
                # 当前卷积计算的区域
                current_input = input[r:r+kernel_size,c:c+kernel_size]
                # 乘法运算
                current_output = current_input * kernel[i]
                conv_sum = np.sum(current_output)

                result[i,r,c] = conv_sum
    
    return result


img = np.array([
       [2,3,7,4,6,2,9],
       [6,6,9,8,7,4,3],
       [3,4,8,3,8,9,7],
       [7,8,3,6,6,3,4],
       [4,2,1,8,3,4,6],
       [3,2,4,1,9,8,3],
       [0,1,3,9,2,1,4]])
kernel = [[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]]
kernel = np.array(kernel)

output = numpy_conv(input=img,kernel=kernel,stride=1)
print(output)


