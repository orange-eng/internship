
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
from sklearn.cluster import KMeans
import math
from copy import deepcopy

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))


def k_means_cpu(weight,n_clusters,init='k-means++',max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1,1)  # single feature，把整个向量拉直
    #print("weight=",weight)
    k_means = KMeans(n_clusters=n_clusters,init=init,n_init=1,max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_            # 输出是一个一维的。和weight的维度一致
    labels = labels.reshape(org_shape)  # reshape与输出相一致

    # 输出centroids聚类中心和labels标签
    return torch.from_numpy(centroids).cuda().view(1,-1),torch.from_numpy(labels).int().cuda()


def reconstruct_weight_from_k_means_result(centroids,labels):
    weight = torch.zeros_like(labels).float().cuda()
    #print("weight_zero=",weight)
    for i,c in enumerate(centroids.cpu().numpy().squeeze()):    

        # squeeze 移除数组中维度为1的维度
        weight[labels==i] = c.item()
    return weight

# 对全连接层的量化过程
class QuantLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True):
        super(QuantLinear,self).__init__(in_features,out_features,bias)
        self.weight_labels = None
        self.bias_labels = None
        self.num_cent = None
        self.quant_flag = None
        self.quant_bias = None

    def kmeans_quant(self,bias=False,quantize_bit=4):   # 把这一层量化成为多少个比特
        self.num_cent = 2 ** quantize_bit
        w = self.weight.data    # 样本中心个数
        centroids, self.weight_labels = k_means_cpu(w.cpu().numpy(),self.num_cent)

        w_q = reconstruct_weight_from_k_means_result(centroids,self.weight_labels)
        self.weight.data = w_q.float()

        self.quant_flag = True
        #self.bias.data = bias

class QuantConv2d(nn.Conv2d):
    def __init__(self,in_features,out_features,kernel_size,stride=1,
                padding=0,dilation=1,groups=1,bias=True):
        super(QuantConv2d,self).__init__(in_features,out_features,
                kernel_size,stride,padding,dilation,groups,bias)
        self.weight_labels = None
        self.bias_labels = None
        self.num_cent = None
        self.quant_flag = None
        self.quant_bias = None
    
    def kmeans_quant(self,bias=False,quantize_bit=4):   # 把这一层量化成为多少个比特
        self.num_cent = 2 ** quantize_bit
        w = self.weight.data    # 样本中心个数
        centroids, self.weight_labels = k_means_cpu(w.cpu().numpy(),self.num_cent)

        w_q = reconstruct_weight_from_k_means_result(centroids,self.weight_labels)
        self.weight.data = w_q.float()

        self.quant_flag = True
        #self.bias.data = bias


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = QuantConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = QuantConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = QuantConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = QuantLinear(7*7*64, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def kmeans_quant(self, bias=False, quantize_bit=4):
        # Should be a less manual way to quantize
        # Leave it for the future
        self.conv1.kmeans_quant(bias, quantize_bit)
        self.conv2.kmeans_quant(bias, quantize_bit)
        self.conv3.kmeans_quant(bias, quantize_bit)
        self.linear1.kmeans_quant(bias, quantize_bit)
    
    def kmeans_update(self):
        self.conv1.kmeans_update()
        self.conv2.kmeans_update()
        self.conv3.kmeans_update()
        self.linear1.kmeans_update()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, total, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)

def main():
    epochs = 2
    batch_size = 64
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path + '/data/MNIST', train=True, download=False,      # 当场下载模型
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path + '/data/MNIST', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = ConvNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        _, acc = test(model, device, test_loader)
    
    quant_model = deepcopy(model)
    print('=='*10)
    print('2 bits quantization')
    quant_model.kmeans_quant(bias=False, quantize_bit=2)
    _, acc = test(quant_model, device, test_loader)
    torch.save(model.state_dict(),"mnist_Q_cnn.pt")
    return model, quant_model

model, quant_model = main()


'''
w = torch.rand(4,5)
print(w)


centroids, labels = k_means_cpu(w,2**4)    # 把这个权值聚类成2类。输出为0或1
print("centroids=",centroids)
print("label=",labels)      # 1代表比较大的数字，0代表比较小的数字（反之亦然）

# weight = reconstruct_weight_from_k_means_result(centroids,labels)
# print(weight)
'''