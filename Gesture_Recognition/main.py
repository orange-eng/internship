import cv2
import picture as pic
 
font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
width, height = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
 
cap = cv2.VideoCapture(0) #开摄像头
 
if __name__ == "__main__":
	while(1):
		ret, frame = cap.read() #读取摄像头的内容
		frame = cv2.flip(frame, 2)
		roi = pic.binaryMask(frame, x0, y0, width, height) #取手势所在框图并进行处理
		key = cv2.waitKey(1) & 0xFF#按键判断并进行一定的调整
		#按'j''l''u''j'分别将选框左移，右移，上移，下移
		#按'q'键退出录像
		if key == ord('i'):
			y0 += 5
		elif key == ord('k'):
			y0 -= 5
		elif key == ord('l'):
			x0 += 5
		elif key == ord('j'):
			x0 -= 5
		if key == ord('q'):
			break
		cv2.imshow('frame', frame) #播放摄像头的内容
	cap.release()
	cv2.destroyAllWindows() #关闭所有窗口
