
import cv2
import numpy as np
 
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0)) #画出截取的手势框图
	roi = frame[y0:y0+height, x0:x0+width] #获取手势框图
	cv2.imshow("roi", roi) #显示手势框图
	res = skinMask(roi) #进行肤色检测
	cv2.imshow("res", res) #显示肤色检测后的图像
	return res
##########方法二###################
########HSV颜色空间H范围筛选法######
def skinMask(roi):
	low = np.array([0, 48, 50]) #最低阈值
	high = np.array([20, 255, 255]) #最高阈值
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #转换到HSV空间
	mask = cv2.inRange(hsv,low,high) #掩膜，不在范围内的设为255
	res = cv2.bitwise_and(roi,roi, mask = mask) #图像与运算
	return res

##########方法三###################
#########椭圆肤色检测模型##########
def skinMask_3(roi):
	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,Cr,Cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(Cr.shape, dtype = np.uint8) #掩膜
	(x,y) = Cr.shape
	for i in range(0, x):
		for j in range(0, y):
			if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #若不在椭圆区间中
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res

################方法四####################
####YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res

##########方法五###################
########Cr，Cb范围筛选法###########
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(cr.shape, dtype = np.uint8)
	(x,y) = cr.shape
	for i in range(0, x):
		for j in range(0, y):
			#每个像素点进行判断
			if(cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res
