


import cv2
import numpy as np
import math

#基于Grabcut算法的前景分割
src = cv2.imread("test_img/4.jpg")
src = cv2.resize(src, (0,0), fx=0.5, fy=0.5)
r = cv2.selectROI('input', src, False)  # 返回 (x_min, y_min, w, h)

roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]# roi区域
mask = np.zeros(src.shape[:2], dtype=np.uint8)# 原图mask
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 矩形roi

bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组
fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组

cv2.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')# 提取前景和可能的前景区域

result = cv2.bitwise_and(src,src,mask=mask2)
cv2.imwrite('forward.png', result)
cv2.imshow("forard", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
