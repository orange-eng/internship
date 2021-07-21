import os
import pandas as pd
import numpy as np

all_file_dir = 'datasets'
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
df = pd.read_csv("csv\\food_data.csv")
image_path_list = df['images'].values
print(image_path_list)

label_list = df['label'].values
# 划分训练集和校验集
all_size = len(image_path_list)
train_size = int(all_size * 0.8)

train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:]
val_label_list = label_list[train_size:]
