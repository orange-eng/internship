import numpy as np
from PIL import Image
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle as pd

class MyDataset(Dataset):
    #继承paddle.io.Dataset类
    def __init__(self,image,label,transform=None):
        # 实现构造函数，定义数据读取格式，划分训练数据集和测试数据集
        super(MyDataset).__init__()
        imgs = image
        labels = label

        self.labels = labels
        self.imgs = imgs
        self.trainsform = transform
    
    def __getitem__(self, idx):  # 按照索引读取每个元素的具体内容 
        fn = self.imgs
        label = self.labels
         # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        for im,la in zip(fn,label):
            img = Image.open(im)
            img = img.convert("RGB")
            img = np.array(img)
            label = np.array([la]).astype(dtype='int64')
        #按照路径读取图片
        if self.trainsform is not None:
            img = self.trainsform(img)
            # 数据标签转化为tensor
        return img,label

    def __len__(self):
        # 返回数据集的长度，多少张图片。
        return len(self.imgs)

