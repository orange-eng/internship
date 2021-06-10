


import numpy as np
import cv2
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))


#鼠标事件的回调函数
def on_mouse(event,x,y,flag,param):        
    global rect
    global leftButtonDowm
    global leftButtonUp
    
    #鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False
        
    #移动鼠标事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and  not leftButtonUp:
            rect[2] = x
            rect[3] = y        
  
    #鼠标左键松开
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDowm and  not leftButtonUp:
            x_min = min(rect[0],rect[2])
            y_min = min(rect[1],rect[3])
            
            x_max = max(rect[0],rect[2])
            y_max = max(rect[1],rect[3])
            
            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDowm = False      
            leftButtonUp = True


img = cv2.imread(path + '/imgs/low_quality1.jpg')
mask = np.zeros(img.shape[:2],np.uint8)


bgdModel = np.zeros((1,65),np.float64)    #背景模型
fgdModel = np.zeros((1,65),np.float64)    #前景模型
rect = [0,0,0,0]                          #设定需要分割的图像范围
    

leftButtonDowm = False                    #鼠标左键按下
leftButtonUp = True                       #鼠标左键松开

cv2.namedWindow('img')                    #指定窗口名来创建窗口
cv2.setMouseCallback('img',on_mouse)      #设置鼠标事件回调函数 来获取鼠标输入
cv2.imshow('img',img)                     #显示图片


while cv2.waitKey(2) == -1:
    #左键按下，画矩阵
    if leftButtonDowm and not leftButtonUp:  
        img_copy = img.copy()
        cv2.rectangle(img_copy,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)  
        cv2.imshow('img',img_copy)
        
    #左键松开，矩形画好 
    elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        rect[2] = rect[2]-rect[0]
        rect[3] = rect[3]-rect[1]
        rect_copy = tuple(rect.copy())   
        rect = [0,0,0,0]
        #物体分割
        cv2.grabCut(img,mask,rect_copy,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
            
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img_show = img*mask2[:,:,np.newaxis]
        #显示图片分割后结果--显示原图
        cv2.imshow('grabcut',img_show)
        cv2.imshow('img',img)    

cv2.waitKey(0)
cv2.destroyAllWindows()
