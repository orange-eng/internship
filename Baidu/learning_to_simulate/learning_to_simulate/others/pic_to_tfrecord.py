
# _*_coding:utf-8_*_

import tensorflow as tf

def write_test(input, output):
    # 借助于TFRecordWriter 才能将信息写入TFRecord 文件
    writer = tf.python_io.TFRecordWriter(output)
    # 读取图片并进行解码
    image = tf.read_file(input)
    image = tf.image.decode_jpeg(image)
    with tf.Session() as sess:
        image = sess.run(image)
        shape = image.shape
        # 将图片转换成string
        image_data = image.tostring()   # 把图像数据转化为二进制数组string存储
        print("img_data=",image)
        #print("img_type=",type(image))
        #print("img_len=",len(image_data))   # 所有像素点的个数
        name = bytes('cat', encoding='utf-8')
        #print("type_name=",type(name))
        print("name=",name)
        # 创建Example对象，并将Feature一一对应填充进去
        example = tf.train.Example(features=tf.train.Features(feature={
             'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
             'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1], shape[2]])),
             'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
        }
        ))
        # 将example序列化成string 类型，然后写入。
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    input_photo = 'train_imgs/4.png'
    output_file = 'tfrecord/4.tfrecord'
    write_test(input_photo, output_file)
