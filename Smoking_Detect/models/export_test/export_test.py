
import argparse

from models.common import *
from utils import google_utils


def imageProcessing(filename, new_shape = (320, 320), color = (114, 114, 114), gray = False):
    img = cv2.imread(filename)

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # Convert
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

    img = img.astype(np.float)

    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndim == 3:
        img = img[np.newaxis ,: ,: ,:]
    elif img.ndim == 2:
        img = img[np.newaxis ,np.newaxis ,: ,:]

    img = np.ascontiguousarray(img)

    return img


# 初始化变量
new_shape = (320, 320)
color = (114, 114, 114)
imgPath = "../../inference/images/smoke1.jpg"   # 准备一张图片

gray = False

# 载入图片
img = imageProcessing(imgPath, new_shape = new_shape, color = color, gray = gray)
img = torch.from_numpy(img)
img = img.float()

# 载入模型
modeldata = torch.load("../../weights/yolov5s.pt", map_location=torch.device('cpu'))
model = modeldata['model'].float().eval()
model = model.float()

# 转化前模型对图片进行处理
pred = model.forward(img)
print(pred[:3]) # 输出pred信息

print("============================================================================================================================") # 输出pred信息
model.model[-1].export = False
traced_script_moudle = torch.jit.trace(model, torch.rand(1,3,new_shape[0],new_shape[1]))
traced_script_moudle.save('yolov5s_torchscript.pt')

jitModel = torch.jit.load('yolov5s_torchscript.pt')
jitPre = jitModel.forward(img)
print(jitPre[:3]) # 相比pred, 丢失了很多信息

