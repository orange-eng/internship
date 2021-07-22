import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np

all_file_dir = 'others\\datasets'
f = open(r'txt\\train.txt','w')
img_list = []
label_list = []
label_id = 0
class_list = [c for c in os.listdir(all_file_dir)]
print(class_list)

for class_dir in class_list:
    image_path_pre = os.path.join(all_file_dir, class_dir)
    for img in os.listdir(image_path_pre):
        print(img)
        f.write("{0}\t{1}\n".format(os.path.join(image_path_pre, img), label_id))
        img_list.append(os.path.join(image_path_pre, img))
        label_list.append(label_id)
    label_id += 1

print("img_list=",img_list)
print("label_list=",label_list)

img_df = pd.DataFrame(img_list)
label_df = pd.DataFrame(label_list)

img_df.columns = ['images']
label_df.columns = ['label']

df = pd.concat([img_df,label_df],axis=1)
print("df=\n",df)

df = df.reindex(np.random.permutation(df.index))

df.to_csv("csv\\food_data.csv",index=0)


# 读取数据
# df = pd.read_csv("learning_to_simulate\\others\\csv\\food_data.csv")
# image_path_list = df['images'].values
# print(image_path_list)

# label_list = df['label'].values
# # 划分训练集和校验集
# all_size = len(image_path_list)
# train_size = int(all_size * 0.8)

# train_image_path_list = image_path_list[:train_size]
# train_label_list = label_list[:train_size]
# val_image_path_list = image_path_list[train_size:]
# val_label_list = label_list[train_size:]


# import numpy as np
# from PIL import Image
# from paddle.io import Dataset
# import paddle.vision.transforms as T
# import paddle

# class MyDataset(Dataset):
#     #继承paddle.io.Dataset类
#     def __init__(self,image,label,transform=None):
#         # 实现构造函数，定义数据读取格式，划分训练数据集和测试数据集
#         super(MyDataset).__init__()
#         imgs = image
#         labels = label

#         self.labels = labels
#         self.imgs = imgs
#         self.trainsform = transform
    
#     def __getitem__(self, idx):  # 按照索引读取每个元素的具体内容 
#         fn = self.imgs
#         label = self.labels
#          # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
#         for im,la in zip(fn,label):
#             img = Image.open(im)
#             img = img.convert("RGB")
#             img = np.array(img)
#             label = np.array([la]).astype(dtype='int64')
#         #按照路径读取图片
#         if self.trainsform is not None:
#             img = self.trainsform(img)
#             # 数据标签转化为tensor
#         return img,label

#     def __len__(self):
#         # 返回数据集的长度，多少张图片。
#         return len(self.imgs)

# transform = T.Compose([
#     T.Resize([224,224]),
#     T.ToTensor(),
#     T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
# ])


# train_dataset = MyDataset(image=image_path_list,label=train_label_list,transform=transform)

# train_loader = paddle.io.DataLoader(train_dataset,places=paddle.CPUPlace(),batch_size=8,shuffle=True)
