
import cv2 as cv
import cv2
import os
import sys
import numpy as np
path = os.path.abspath(os.path.dirname(sys.argv[0]))
def fac_detect_demo(img):
    sh = img.shape
    hight,width = sh[0],sh[1]
    print("w={},h={}".format(width,hight))
    #将图片转化为灰度
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载数据路径
    face_detector = cv.CascadeClassifier(path + "/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray)
    #画出一个mask，遮盖人脸区域
    for x,y,w,h in faces:
#        cv.rectangle(img,(x,y-int(0.2*h)),(x+w,y+int(h)),color=(0,0,0),thickness=-1)
        cv.rectangle(img,(x-int(0.2*w),y-int(0.4*h)),(x+int(1.2*w),y+int(h)),color=(0,0,0),thickness=-1)
    cv.rectangle(img,(0,y+h),(width,hight),color=(0,0,0),thickness=-1)

    cv.imshow("result",img)
    cv.imwrite(path + "/result/result6.jpg",img)
    
#img = cv.imread(path + "/img/picture1.jpg")
#cv.imshow("img",img)
def get_brightness(in_img):
    hsv = cv.cvtColor(in_img, cv.COLOR_BGR2HSV)
    channels = cv.split(hsv)
    v_channel = channels[2]
    print("v_channel_size:",v_channel.shape)
    cv.imshow("v_channel",v_channel)

    h, w = in_img.shape[0], in_img.shape[1]
    total_v = np.sum(v_channel)
    avg_v = total_v / (h * w)
    return avg_v

def func(img):
    #把图片转换为灰度图
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #获取灰度图矩阵的行数和列数
    r,c = gray_img.shape[:2]
    dark_sum=0              #偏暗的像素 初始化为0个
    dark_prop=0             #偏暗像素所占比例初始化为0
    piexs_sum=r*c           #整个灰度图的像素个数为r*c
    pixel_0 = 0
    #遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum<40:    #人为设置的超参数,表示0~39的灰度值为暗
                if colum > 0:
                    dark_sum+=1
                else:
                    pixel_0 += 1
    dark_prop=dark_sum/(piexs_sum - pixel_0)
    print("dark_sum:"+str(dark_sum))
    print("piexs_sum:"+str(piexs_sum - pixel_0))
    print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop))
    if dark_prop >=0.40:    #人为设置的超参数:表示若偏暗像素所占比例超过0.40,则这张图被认为整体环境黑暗的图片
        print("it is dark!")
    else:
        print("it is bright!")


# img = cv.imread(path + "/img/picture6.jpg")
# fac_detect_demo(img)
result = cv.imread(path + "/result/result.jpg")
brightness = get_brightness(result)
print(brightness)
func(result)

cv.waitKey(0)
cv.destroyAllWindows()
