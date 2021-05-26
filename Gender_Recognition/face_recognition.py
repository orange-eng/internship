import cv2 as cv
import numpy as np

def face_detect_demo(src):#人脸检测函数
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)#把图片变成灰度图片，因为人脸的特征需要在灰度图像中查找
    #以下分别是HAAR和LBP特征数据，任意选择一种即可，注意：路径中的‘/’和‘\’是有要求的
    # 通过级联检测器 cv.CascadeClassifier，加载特征数据
    # face_detector = cv.CascadeClassifier("D:/pyproject/cv_renlianjiance/haarcascades/haarcascade_frontalface_alt_tree.xml")
    face_detector = cv.CascadeClassifier(
        "face_shape/data/lbpcascades/lbpcascade_frontalcatface.xml")
    #在尺度空间对图片进行人脸检测，第一个参数是哪个图片，第二个参数是向上或向下的尺度变化，是原来尺度的1.02倍，第三个参数是在相邻的几个人脸检测矩形框内出现就认定成人脸，这里是在相邻的5个人脸检测框内出现，如果图片比较模糊的话建议降低一点
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:#绘制结果图
        #rectangle参数说明，要绘制的目标图像，矩形的第一个顶点，矩形对角线上的另一个顶点，线条的颜色，线条的宽度
        cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow("result", src)#输出结果图

src = cv.imread("images/test_man.png")#图片是JPG和png都可以
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)#创建绘图窗口
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
face_detect_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()#作用是能正常关闭绘图窗口
