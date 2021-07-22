import base64
from aip import AipFace
import cv2
import os

# 配置百度aip参数
APP_ID = '19484855'
API_KEY = 'V2mDOleCsk3yEE6P5MgVwSjI'
SECRET_KEY = 'RbRMAuPmz8QpDweikrbpfGQjXUm7HiCD'
a_face = AipFace(APP_ID, API_KEY, SECRET_KEY)
image_type = 'BASE64'

options = {'face_field': 'age,gender,beauty', "max_face_num": 10}
max_face_num = 10

def get_file_content(file_path):
    """获取文件内容"""
    with open(file_path, 'rb') as fr:
        content = base64.b64encode(fr.read())
        return content.decode('utf8')

def face_score(file_path):
    """脸部识别分数"""
    result = a_face.detect(get_file_content(file_path), image_type, options)
    return result

path = "img/5gender"
pic_name = os.listdir(path)


num = 0
for i in range(len(pic_name)):
    img_name = path + "/" + pic_name[i]
    # 图片地址，图片与程序同一目录下
    file_path = img_name
    print(file_path)
    result = face_score(file_path)
    print(result)
    # #从文件读取图像并转为灰度图像
    img = cv2.imread(file_path)
    # 图片放文字
    # 设置文件的位置、字体、颜色等参数
    font = cv2.FONT_HERSHEY_DUPLEX
    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
    color = (0, 0, 255)
    if result['result'] != None:
        for item in result['result']['face_list']:
            x = int(item['location']['left'])
            y = int(item['location']['top'])
            w = item['location']['width']
            h = item['location']['height']
            age = item['age']
            beauty = item['beauty']
            gender = item['gender']['type']
            if gender == 'female':
                num = num + 1
                print(num)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # cv2.putText(img, 'age:%s' % age, (x, y + h + 10), font, 1, color, 1)
            # cv2.putText(img, 'beauty:%s' % beauty, (x, y + h + 30), font, 1, color, 1)
            # cv2.putText(img, 'gender:%s' % gender, (x, y + h + 50), font, 1, color, 1)
# print(num)

    #cv2.imshow('Image', img)
    # 按任意键退出
    # key = cv2.waitKey()
    # if key == 27:
    #     # 销毁所有窗口
    #     cv2.destroyAllWindows()


# img_name = 'img/5gender/image204.png'
# # 图片地址，图片与程序同一目录下
# file_path = img_name
# print(file_path)
# result = face_score(file_path)
# print(result)
# #从文件读取图像并转为灰度图像
# img = cv2.imread(file_path)
# # 图片放文字
# # 设置文件的位置、字体、颜色等参数
# font = cv2.FONT_HERSHEY_DUPLEX
# # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
# color = (0, 0, 255)
# num = 0
# if result['result'] != None:
#     for item in result['result']['face_list']:
#         x = int(item['location']['left'])
#         y = int(item['location']['top'])
#         w = item['location']['width']
#         h = item['location']['height']
#         age = item['age']
#         beauty = item['beauty']
#         gender = item['gender']['type']

