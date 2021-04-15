# Bright_Dark_Discrimination

## bright_dark_photo_1
- 使用HSV通道分离图像，然后提取V通道（亮度值），求取整张图片像素点的亮度平均值brightness。此时可以设置一个阈值threshold，如果brightness>threshold则认为是白天图片，否则为黑夜图片
```python
def get_brightness(in_img):
    hsv = cv.cvtColor(in_img, cv.COLOR_BGR2HSV)
    channels = cv.split(hsv)
    v_channel = channels[2]
    h, w = in_img.shape[0], in_img.shape[1]
    total_v = np.sum(v_channel)
    avg_v = total_v / (h * w)
    return avg_v
```

## bright_dark_photo_2
- 使用灰度图像，并设置一个阈值threshold，如果像素值<threshold则认为该像素为“暗像素”，如果“暗像素”个数过多，即可认为该图片为黑夜。
```python
def func(img):
    #把图片转换为灰度图
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #获取灰度图矩阵的行数和列数
    r,c = gray_img.shape[:2]
    dark_sum=0              #偏暗的像素 初始化为0个
    dark_prop=0             #偏暗像素所占比例初始化为0
    piexs_sum=r*c           #整个弧度图的像素个数为r*c
    #遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum<40:    #人为设置的超参数,表示0~39的灰度值为暗
                dark_sum+=1
                dark_prop=dark_sum/(piexs_sum)
    print("dark_sum:"+str(dark_sum))
    print("piexs_sum:"+str(piexs_sum))
    print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop))
    if dark_prop >=0.40:    #人为设置的超参数:表示若偏暗像素所占比例超过0.40,则这张图被认为整体环境黑暗的图片
        print("it is dark!")
    else:
        print("it is bright!")
```

## bright_dark_video
- 判别某个视频是否为白天或是黑夜。对每一帧图像进行像素点统计，再使用以上两种方法判断视频是否为白天或是黑夜。
