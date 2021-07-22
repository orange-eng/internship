#导入OpenCV
import cv2
#创建一个VideoCapture对象，它的参数可以是设备索引或视频文件的名称（下面会讲到）。设备索引只是指定哪台摄像机的号码。0代表第一台摄像机、1代表第二台摄像机。之后，可以逐帧捕捉视频。最后释放捕获。
cap = cv2.VideoCapture(0)
while True:
    #读取帧
    ret,frame = cap.read()
    #将视频灰度化
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #将视频灰度化显示
    cv2.imshow('frame',frame)
    #按‘q'退出
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
#释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
