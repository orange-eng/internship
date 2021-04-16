
import cv2 as cv
import cv2
import math
import random
import os
import sys
import numpy as np
path = os.path.abspath(os.path.dirname(sys.argv[0]))

def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
    	for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
    	for y in range(1, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def gasuss_noise(image, mean=0, var=0.002):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

img = cv.imread(path + "/img/picture1.jpg")
noise_img = gasuss_noise(img)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
noise_img = cv.cvtColor(noise_img,cv.COLOR_BGR2GRAY)
# cv.imshow("img",noise_img)
# cv.waitKey(0)
# cv.imwrite(path + "/img/picture1_nosie.jpg",noise_img)
flag = 4
if flag == 0:
    out = entropy(img)
    noise_out = entropy(noise_img)
elif flag == 1:
    out = Vollath(img)
    noise_out = Vollath(noise_img)
elif flag == 2:
    out = energy(img)
    noise_out = energy(noise_img)
elif flag == 3:
    out = SMD2(img)
    noise_out = SMD2(noise_img)
elif flag == 4:
    out = brenner(img)
    noise_out = brenner(noise_img)
   
print("normal_img:",out)
print("noise__img:",noise_out)
