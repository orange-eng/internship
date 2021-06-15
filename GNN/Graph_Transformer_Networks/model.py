import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):  # layers多个GTLayer组成的; 多头channels
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))  # 第一个GT层,edge类别构建的矩阵
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))  # GCN
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):  # 自己写了一个GCN
        X = torch.mm(X, self.weight)  # X-features; self.weight-weight
        H = self.norm(H, add=True)  # H-第i个channel下邻接矩阵;
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)  # Q1
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)  # Q2
        return H_

    def norm(self, H, add=False):
        H = H.t()   # t
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))  # 建立一个对角阵; 除了自身节点，对应位置相乘。Degree(排除本身)
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)  # 按行求和, 即每个节点的dgree的和
        deg_inv = deg.pow(-1)  # deg-1 归一化操作
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)  # 转换成n*n的矩阵
        H = torch.mm(deg_inv,H)  # 矩阵内积
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2)   # A.unsqueeze(0)=[1,N,N,edgeType]=>[1,edgeType,N,N]; 卷积输出的channel数量
        Ws = []
        for i in range(self.num_layers):  # 两层GTLayer:{edgeType}
            if i == 0:
                H, W = self.layers[i](A)  # GTN0:两层GTConv; A:edgeType的邻接矩阵; output: H(A(l)), W:归一化的Conv
            else:
                H = self.normalization(H)   # Conv矩阵，D-1*A的操作
                H, W = self.layers[i](A, H)  # 第一层计算完了A(原始矩阵), H(上一次计算后的A(l)); output: A2, W(第二层Conv1)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):   # conv的channel数量
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))  # X-features; H[i]-第i个channel输出的邻接矩阵Al[i]; gcn_conv:Linear
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)  # X_拼接之后输出
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    # 不同edge类型的组合
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # 1x1卷积的channel数量
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)  # W1
            self.conv2 = GTConv(in_channels, out_channels)  # W2
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):  # A:[1,edgeType,N,N]
        if self.first == True:
            a = self.conv1(A)  # GTConv=>[2, N, N] #Q1
            b = self.conv2(A)  # Q2
            #*** 作了第一次矩阵相乘，得到A1
            H = torch.bmm(a,b)  # torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]  # conv-softmax: 是为了下一次直接使用吗？
        else:
            a = self.conv1(A)  # 第二层只有一个conv1; output:Conv输出归一化edge后的结果
            H = torch.bmm(H_,a)  # H_上一层的输出矩阵A1; 输出这一层后的结果A2;
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W   # H = A(1) ... A(l); W = 归一化后的权重矩阵

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))  #
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):  # self.weight:带有channel的conv;
        '''
        0) 对weight(conv)进行softmax
        1) 对每个节点在每个edgeType上进行[2, 5, 1, 1]的卷积操作;
        2) 对每个edgeType进行加权求和，加权是通过0)softmax
        '''
        # F.softmax(self.weight, dim=1) 对self.weight做softmax:[2, 5, 1, 1]
        # A: [1, 5, 8994, 8994]:带有edgeType的邻接矩阵
        # [1, 5, 8994, 8994]*[2, 5, 1, 1] => [2, 5, 8994, 8994]
        # sum:[2, 8994, 8994]
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
