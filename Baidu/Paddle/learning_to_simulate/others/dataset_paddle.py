import paddle
from paddle.io import Dataset
BATCH_SIZE = 64
BATCH_NUM = 20
IMAGE_SIZE = (28, 28)
CLASS_NUM = 10
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, num_samples):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.num_samples

# 测试定义的数据集
custom_dataset = MyDataset(10)
print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break