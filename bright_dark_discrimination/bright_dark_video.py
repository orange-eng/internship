import cv2
import numpy as np
import os,time
import shutil

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

def GetImgNameByEveryDir(file_dir,videoProperty):  
    FileNameWithPath = [] 
    FileName         = []
    FileDir          = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] in videoProperty:  
                FileNameWithPath.append(os.path.join(root, file))  # 保存图片路径
                FileName.append(file)                              # 保存图片名称
                FileDir.append(root[len(file_dir):])               # 保存图片所在文件夹
    return FileName,FileNameWithPath,FileDir
 
def img_to_GRAY(img,pic_path):
    #把图片转换为灰度图
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #获取灰度图矩阵的行数和列数
    r,c = gray_img.shape[:2]
    piexs_sum=r*c #整个图的像素个数
    #遍历灰度图的所有像素
    #灰度值小于60被认为是黑
    dark_points = (gray_img < 60)
    target_array = gray_img[dark_points]
    dark_sum = target_array.size #偏暗的像素
    dark_prop=dark_sum/(piexs_sum) #偏暗像素所占比例
    if dark_prop >=0.60: #若偏暗像素所占比例超过0.6,认为为整体环境黑暗的图片
        return 1
    else:
        return 0

if __name__ =='__main__':
    path=path + "//video"
    new_path=path
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    FileName,FileNameWithPath,FileDir=GetImgNameByEveryDir(path,'.mp4')
    for i in range(len(FileNameWithPath)):
        video_capture = cv2.VideoCapture(FileNameWithPath[i])
        video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(video_capture.get(5))
        start_fps=2*video_fps #从2秒开始筛选
        end_fps=6*video_fps #6秒结束
        avg_fps=end_fps-start_fps #总共fps
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_fps) #设置视频起点
        new_paths=new_path+"\\"+FileName[i]
        j=0
        count=0
        while True:
            success,frame = video_capture.read()
            if success:
                j += 1
                if(j>=start_fps and j <= end_fps):
                    flag=img_to_GRAY(frame,FileNameWithPath[i])
                    if flag==1:
                        count+=1
                elif(j>end_fps):
                    break
            else:
                break
        if count>int(avg_fps*0.48): #大于fps50%为黑夜
            print("%s,该视频为黑夜"%FileNameWithPath[i])
            video_capture.release() #释放读取的视频，不占用视频文件
            time.sleep(0.2)
            shutil.move(FileNameWithPath[i],new_paths)
        else:
            print("%s,该视频为白天"%FileNameWithPath[i])
