import cv2
import numpy as np
import matplotlib.pyplot as plt


class Detector(object):

    def __init__(self, name='my_video'):

        self.name = name

        self.threshold = 40

        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

    def catch_video(self, video_index=0):

        # cv2.namedWindow(name)

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_num = 0

        while cap.isOpened():

            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:

                raise Exception('Error.')

            if not frame_num:
                # 这里是由于第一帧图片没有前一帧
                previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换成灰度图

            gray = cv2.absdiff(gray, previous)  # 计算绝对值差

            gray = cv2.medianBlur(gray, 3)  # 中值滤波

            ret, mask = cv2.threshold(
                gray, self.threshold, 255, cv2.THRESH_BINARY)


            mask = cv2.erode(mask, self.es, iterations=1)
            mask = cv2.dilate(mask, self.es, iterations=1)

            cv2.imshow(self.name, frame)  # 在window上显示图片
            cv2.imshow(self.name+'_frame', mask)  # 边界

            previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_num += 1

            key = cv2.waitKey(10)

            if key & 0xFF == ord('q'):
                # 按q退出
                break

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

            if cv2.getWindowProperty(self.name+'_frame', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = Detector()

    detector.catch_video()
