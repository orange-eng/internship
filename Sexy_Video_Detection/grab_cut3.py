


import cv2
import numpy as np


def readimg(src_path,background_path):
	src = cv2.imread(src_path)
	background = cv2.imread(background_path)
	h, w, ch = src.shape
	mask = np.zeros(src.shape[:2], dtype=np.uint8)
	rect = (53,12,w-100,h-12)
	bgdmodel = np.zeros((1,65),np.float64)
	fgdmodel = np.zeros((1,65),np.float64)

	cv2.grabCut(src,mask,rect,bgdmodel,fgdmodel,5,mode=cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
	object = cv2.bitwise_and(src, src, mask=mask2)
	cv2.imshow("object", object)

	# 高斯模糊
	se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	cv2.dilate(mask2, se, mask2)
	mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
	cv2.imshow('mask',mask2)

	# 虚化背景
	background = cv2.GaussianBlur(background, (0, 0), 15)

	# blend image
	result = np.zeros((h, w, ch), dtype=np.uint8)
	for row in range(h):
		for col in range(w):
			w1 = mask2[row, col] / 255.0
			b, g, r = src[row, col]
			b1,g1,r1 = background[row, col]
			b = (1.0-w1) * b1 + b * w1
			g = (1.0-w1) * g1 + g * w1
			r = (1.0-w1) * r1 + r * w1
			result[row, col] = (b, g, r)
	return result

if __name__ == '__main__':
	src_path = './imgs/5.jpg'
	background_path = './imgs/1.jpg'
	result =  readimg(src_path,background_path)
	cv2.imshow("result", result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
