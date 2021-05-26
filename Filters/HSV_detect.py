# -*- coding:utf-8 -*-
#本程序用于将一张彩色图片分解成BGR的分量显示，灰度图显示，HSV分量显示
import cv2  #导入opencv模块
import numpy as np
import os 


def average_pixel(img_name):
    img = cv2.imread(img_name)  #导入图片，图片放在程序所在目录
    #使用cvtColor转换为HSV图
    out_img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)#将图片转换为灰度图
    hsvChannels=cv2.split(out_img_HSV)  #将HSV格式的图片分解为3个通道

    img_v = hsvChannels[2]

    height,width = img_v.shape

    square = 50
    padding = 50

    left_up = img_v[padding:padding+square,padding:padding+square]
    right_up = img_v[padding:padding+square,width-padding-square:width-padding]
    left_down = img_v[height-padding-square:height-padding,padding:padding+square]
    right_down = img_v[height-padding-square:height-padding,width-padding-square:width-padding]

    num = 0
    average = 0

    name = [left_up,right_up,left_down,right_down]
    for k in range(4):     
        pixel = name[k]
        for i in range(square):
            for j in range(square):
                num = num + pixel[i,j]

    average = num/(4*square*square)

    if average >= 125:
        return 1
    else:
        return 0

path = "img/GOOD_CASE"
pic_name = os.listdir(path)
# #print(pic_name)
flag = 0
flag_num = 0 

for i in range(len(pic_name)):
    img_name = path + "/" + pic_name[i]
    flag = average_pixel(img_name)
    if flag == 1:
        flag_num = flag_num + 1

print(flag_num)

# GOOD_CASE threshold=125   50:13
# jump threshold=125    8:61
