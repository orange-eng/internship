import cv2 as cv
import numpy as np
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
# def face_detect_demo(image):
#     sh = image.shape
#     w,h = sh[0],sh[1]
#     print("w={},h={}".format(w,h))
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     # face_detector = cv.CascadeClassifier("D:/pyproject/cv_renlianjiance/haarcascades/haarcascade_frontalface_alt_tree.xml")
#     face_detector = cv.CascadeClassifier(path + "/haarcascades/haarcascade_frontalface_default.xml")
#     faces = face_detector.detectMultiScale(gray)
#     for x, y, w, h in faces:
#         cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     font=cv.FONT_HERSHEY_SIMPLEX#使用默认字体
    
#     image=cv.putText(image,"3",(0,40),font,1.2,(255,255,255),2)
#     cv.imshow("result", image)
def fac_detect_demo(img):
    sh = img.shape
    hight,width = sh[0],sh[1]
    #print("w={},h={}".format(width,hight))
    #将图片转化为灰度
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载数据路径
    face_detector = cv.CascadeClassifier(path + "/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray)
    #画出一个mask，遮盖人脸区域
    edge = 0        #用于记录下半身的坐标
    for x,y,w,h in faces:
        cv.rectangle(img,(x-int(0.2*w),y-int(0.4*h)),(x+int(1.2*w),y+int(h)),color=(0,0,0),thickness=-1)
        edge = y+h  #记录下半身的坐标
    cv.rectangle(img,(0,edge),(width,hight),color=(0,0,0),thickness=-1)
    cv.imshow("result",img)


capture = cv.VideoCapture(path + "/video/4.mp4")

cv.namedWindow("result", cv.WINDOW_AUTOSIZE)


#-----------------------------------------------
##########检测图片
#-----------------------------------------------


#-----------------------------------------------
##########检测视频
#-----------------------------------------------
while (True):
    #按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    ret, frame = capture.read()
    fac_detect_demo(frame)
    c = cv.waitKey(1)
    if c == 27:#当键盘按下‘ESC’退出程序
        break

cv.waitKey(0)
cv.destroyAllWindows()#作用是能正常关闭绘图窗口
