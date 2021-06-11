'''
Extract panel :kmeans聚类
'''
import cv2
import numpy as np
import math
def panelAbstract(srcImage):
    #   read pic shape
    imgHeight,imgWidth = srcImage.shape[:2]
    imgHeight = int(imgHeight);imgWidth = int(imgWidth)
    # 均值聚类提取前景:二维转一维
    imgVec = np.float32(srcImage.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS 
    ret,label,clusCenter = cv2.kmeans(imgVec,2,None,criteria,10,flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape((srcImage.shape))
    imgres = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)
    bwThresh = int((np.max(imgres)+np.min(imgres))/2)
    _,thresh = cv2.threshold(imgres,bwThresh,255,cv2.THRESH_BINARY_INV)
    threshRotate = cv2.merge([thresh,thresh,thresh])
    # 确定前景外接矩形
    #find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight,imgWidth]);maxvalx = 0
    minvaly = np.max([imgHeight,imgWidth]);maxvaly = 0
    maxconArea = 0;maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i
    objCont = contours[maxAreaPos]
    # 旋转校正前景
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly,objCont[j][0][0]])
        maxvaly = np.max([maxvaly,objCont[j][0][0]])
        minvalx = np.min([minvalx,objCont[j][0][1]])
        maxvalx = np.max([maxvalx,objCont[j][0][1]])
    if rect[2] <=-45:
        rotAgl = 90 +rect[2]
    else:
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx,minvaly:maxvaly,:]
    else:
        rotCtr = rect[0]
        rotCtr = (int(rotCtr[0]),int(rotCtr[1]))
        rotMdl = cv2.getRotationMatrix2D(rotCtr,rotAgl,1)
        imgHeight,imgWidth = srcImage.shape[:2]
        #图像的旋转
        dstHeight = math.sqrt(imgWidth *imgWidth + imgHeight*imgHeight)
        dstRotimg = cv2.warpAffine(threshRotate,rotMdl,(int(dstHeight),int(dstHeight)))
        dstImage = cv2.warpAffine(srcImage,rotMdl,(int(dstHeight),int(dstHeight)))
        dstRotimg = cv2.cvtColor(dstRotimg,cv2.COLOR_BGR2GRAY)
        _,dstRotBW = cv2.threshold(dstRotimg,127,255,0)
        contours, hierarchy = cv2.findContours(dstRotBW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0;maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x,y,w,h = cv2.boundingRect(contours[maxAreaPos])
        #提取前景：panel
        panelImg = dstImage[int(y):int(y+h),int(x):int(x+w),:]

    return panelImg

if __name__=="__main__":
   srcImage = cv2.imread('test_img/4.jpg')
   a=panelAbstract(srcImage)
   cv2.imshow('figa',a)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
