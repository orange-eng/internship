import numpy as np
import cv2
from matplotlib import pyplot as plt

#创建一个VideoCapture对象，它的参数可以是设备索引或视频文件的名称（下面会讲到）。设备索引只是指定哪台摄像机的号码。0代表第一台摄像机、1代表第二台摄像机。之后，可以逐帧捕捉视频。最后释放捕获。
# cap = cv2.VideoCapture(0)
# while True:
#     #读取帧
#     ret,img = cap.read()
#     mask = np.zeros(img.shape[:2],np.uint8)
#     bgdModel = np.zeros((1,65),np.float64)
#     fgdModel = np.zeros((1,65),np.float64)
#     rect = (50,50,450,290)
#     cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#     img = img*mask2[:,:,np.newaxis]
#     #将视频灰度化显示
#     cv2.imshow('frame',img)
#     #按‘q'退出
#     if cv2.waitKey(1) and 0xFF == ord('q'):
#         break
# #释放资源并关闭窗口
# cap.release()
# cv2.destroyAllWindows()

img = cv2.imread('imgs/4.jpg')
mask = np.zeros(img.shape[:2],np.uint8)


bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow('frame',img)
cv2.waitKey(0)
#plt.imshow(img),plt.colorbar(),plt.show()
