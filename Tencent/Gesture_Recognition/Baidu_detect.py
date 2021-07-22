# 使用百度云内置的APP完成手势识别， 链接在这里https://www.easck.com/cos/2020/0728/562711.shtml

import os
import cv2
from aip import AipBodyAnalysis
from aip import AipSpeech
from threading import Thread
import time
from playsound import playsound
""" 你的 APPID AK SK """

APP_ID = '24036624'
API_KEY = 'tqUXUxLV0Gl6ON29N0cSqgfP'
SECRET_KEY =  '6UW7qfpP9z0xYw0G2OmKUuL27NeifZ6m'
''' 调用'''

hand={'One':'数字1','Five':'数字5','Fist':'拳头','Ok':'OK',
      'Prayer':'祈祷','Congratulation':'作揖','Honour':'作别',
      'Heart_single':'比心心','Thumb_up':'点赞','Thumb_down':'Diss',
      'ILY':'我爱你','Palm_up':'掌心向上','Heart_1':'双手比心1',
      'Heart_2':'双手比心2','Heart_3':'双手比心3','Two':'数字2',
      'Three':'数字3','Four':'数字4','Six':'数字6','Seven':'数字7',
      'Eight':'数字8','Nine':'数字9','Rock':'Rock','Insult':'竖中指','Face':'脸'}

#语音合成q
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

#手势识别
gesture_client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

capture = cv2.VideoCapture(0)#0为默认摄像头
def camera():

    while True:
        #获得图片
        ret, frame = capture.read()
        # cv2.imshow(窗口名称, 窗口显示的图像)
        #显示图片
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
Thread(target=camera).start()#引入线程防止在识别的时候卡死

def gesture_recognition():

    #第一个参数ret 为True 或者False,代表有没有读取到图片

    #第二个参数frame表示截取到一帧的图片

    while True:
        try:
            ret, frame = capture.read()

            #图片格式转换
            image = cv2.imencode('.jpg',frame)[1]

            gesture =  gesture_client.gesture(image)   #AipBodyAnalysis内部函数
            words = gesture['result'][0]['classname']

            voice(hand[words])
            print(hand[words])

        except:
            voice('识别失败')
        if cv2.waitKey(1) == ord('q'):
            break

def voice(words):
    #语音函数
    result  = client.synthesis(words, 'zh', 1, {
        'vol': 5,
    })
    if not isinstance(result, dict):
        with open('./res.mp3', 'wb') as f:
            f.write(result)
            f.close()
        playsound('./res.mp3')

gesture_recognition()