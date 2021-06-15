import numpy as np
import pandas as pd
import networkx as nx
import torch
import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)
print(torch.cuda.is_available())
'''
文档：https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
安装 * torch >= 1.6.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-scatter
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html    
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-sparse
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-cluster
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-spline-conv
pip install torch-geometric



'''
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)






# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv,GATConv,SAGEConv
# from torch_geometric.datasets import Planetoid
#
# '''
# 下载报错，将所有data文件下载到本地
# https://github.com/kimiyoung/planetoid
# 将cora相关文件放入到raw文件中
# '''
# dataset = Planetoid(root='./tmp/Cora',name='Cora')
#
#
# class GCN(torch.nn.Module):
#     def __init__(self,feature, hidden, classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(feature,hidden)
#         self.conv2 = GCNConv(hidden, classes)
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
# # data = dataset[0].to(device)
# # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# #
# # model.train()
# # for epoch in range(200):
# #     optimizer.zero_grad()
# #     out = model(data)
# #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
# #     loss.backward()
# #     optimizer.step()
# #     # print(epoch,loss.item())
# #
# # model.eval()
# # _, pred = model(data).max(dim=1)
# # correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# # acc = int(correct) / int(data.test_mask.sum())
# # print(acc)
#
#
# class GAT(torch.nn.Module):
#     def __init__(self, feature, hidden, classes, heads=1):
#         super(GAT,self).__init__()
#         self.gat1 = GATConv(feature, hidden, heads=heads)
#         self.gat2 = GATConv(hidden*heads, classes)
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.gat1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.gat2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = GAT(dataset.num_node_features, 8, dataset.num_classes,heads=4).to(device)
# # data = dataset[0].to(device)
# # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# #
# # model.train()
# # for epoch in range(200):
# #     optimizer.zero_grad()
# #     out = model(data)
# #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
# #     loss.backward()
# #     optimizer.step()
# #     # print(epoch,loss.item())
# #
# # model.eval()
# # _, pred = model(data).max(dim=1)
# # correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# # acc = int(correct) / int(data.test_mask.sum())
# # print(acc)
#
#
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, feature, hidden, classes):
#         super(GraphSAGE, self).__init__()
#         self.sage1 = SAGEConv(feature, hidden)
#         self.sage2 = SAGEConv(hidden, classes)
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.sage1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.sage2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GraphSAGE(dataset.num_node_features, 8, dataset.num_classes).to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     # print(epoch,loss.item())
#
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(acc)