

# _*_coding:utf-8_*_

# 将图片保存成TFRecords
import os
import tensorflow as tf
from PIL import Image
import random
import cv2
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_image(filename, resize_height, resize_width, normalization=False):
    '''
        读取图片数据，默认返回的是uint8, [0, 255]
        :param filename:
        :param resize_height:
        :param resize_width:
        :param normalization:  是否归一化到 [0.0, 1.0]
        :return:  返回的图片数据
        '''

    bgr_image = cv2.imread(filename)
    # print(type(bgr_image))
    # 若是灰度图则转化为三通道
    if len(bgr_image.shape) == 2:
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    # 将BGR转化为RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # show_image(filename, rgb_image)
    # rgb_image=Image.open(filename)
    if resize_width > 0 and resize_height > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image

def load_labels_file(filename, labels_num=1, shuffle=False):
    '''
        载图txt文件，文件中每行为一个图片信息，且以空格隔开，图像路径 标签1  标签2
        如  test_image/1.jpg 0 2
        :param filename:
        :param labels_num:  labels个数
        :param shuffle: 是否打乱顺序
        :return:  images type-> list
        :return：labels type->lis\t
        '''
    images = []
    labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        # print(lines_list)  # ['plane\\0499.jpg 4\n', 'plane\\0500.jpg 4\n']
        if shuffle:
            random.shuffle(lines_list)
        for lines in lines_list:
            line = lines.rstrip().split(" ")  # rstrip 删除 string 字符串末尾的空格.  ['plane\\0006.jpg', '4']
            label = []
            for i in range(labels_num):  # labels_num 1      0 1所以i只能取1
                label.append(int(line[i + 1]))  # 确保读取的是列表的第二个元素
            # print(label)
            images.append(line[0])
            # labels.append(line[1])  # ['0', '4']
            labels.append(label)
    # print(images)
    # print(labels)
    return images, labels


def create_records(image_dir, file, output_record_dir, resize_height, resize_width, shuffle, log=5):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件，仅获取一个label
    images_list, labels_list = load_labels_file(file, 1, shuffle)
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path = os.path.join(image_dir, images_list[i])
        if not os.path.exists(image_path):
            print("Error:no image", image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()

        if i % log == 0 or i == len(images_list) - 1:
            print("-----------processing:%d--th------------" % (i))
            print('current image_path=%s' % (image_path), 'shape:{}'.format(image.shape),
                  'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label = labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def get_example_nums(tf_records_filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

if __name__ == '__main__':
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    shuffle = True
    log = 5
    image_dir = 'train_imgs'
    train_labels = 'train.txt'
    train_record_output = 'train.tfrecord'
    create_records(image_dir, train_labels, train_record_output, resize_height, resize_width, shuffle, log)
    train_nums = get_example_nums(train_record_output)
    print("save train example nums={}".format(train_nums))