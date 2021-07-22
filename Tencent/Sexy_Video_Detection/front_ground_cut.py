import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('imgs/2.png')
mask = np.zeros(img.shape[:2], np.uint8)

# zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (1, 1, img.shape[1], img.shape[0])
# 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
# img - 输入图像
# mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
# rect - 包含前景的矩形，格式为 (x,y,w,h)
# bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
# iterCount - 算法的迭代次数
# mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

# np.where 函数是三元表达式 x if condition else y的矢量化版本
# result = np.where(cond,xarr,yarr)
# 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
# | 是逻辑运算符or的另一种表现形式
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# mask2[:, :, np.newaxis] 增加维度
img = img * mask2[:, :, np.newaxis]

# 显示图片
plt.subplot(121), plt.imshow(img)
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('imgs/2.png'), cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
plt.show()
