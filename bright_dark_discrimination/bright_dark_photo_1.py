import logging
import cv2 as cv
import numpy as np
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

def get_brightness(in_img):
    hsv = cv.cvtColor(in_img, cv.COLOR_BGR2HSV)
    channels = cv.split(hsv)
    v_channel = channels[2]
    h, w = in_img.shape[0], in_img.shape[1]
    total_v = np.sum(v_channel)
    avg_v = total_v / (h * w)
    return avg_v

logging.basicConfig(level=logging.INFO)

img = path + "/img/dark.png"
img = cv.imread(img, cv.IMREAD_COLOR)
brightness = get_brightness(img)
logging.info("day 的平均亮度：{}".format(brightness))
cv.imshow("img", img)
cv.waitKey(0)
if brightness > 100:
    print("This photo is bright!")
else:
    print("This photo is dark")