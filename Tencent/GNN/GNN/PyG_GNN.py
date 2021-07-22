from torch_geometric.datasets import Planetoid

# '''
# 下载报错，将所有data文件下载到本地
# https://github.com/kimiyoung/planetoid
# 将cora相关文件放入到raw文件中
# '''

dataset = Planetoid(root='./tmp/Cora',name='Cora')
print((dataset[0].train_mask).sum())
print((dataset[0].test_mask).sum())
print((dataset[0].val_mask).sum())
print(dataset[0])

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

# class GCN_Net(torch.nn.Module):
#     def __init__(self, features, hidden, classes):
#         super(GCN_Net, self).__init__()
#         self.conv1 = GCNConv(features, hidden)
#         self.conv2 = GCNConv(hidden, classes)
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
# data = dataset[0]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# acc = int(correct)/ int(data.test_mask.sum())
# print('GCN:',acc)
#
# class GraphSAGE_Net(torch.nn.Module):
#     def __init__(self, features, hidden, classes):
#         super(GraphSAGE_Net, self).__init__()
#         self.sage1 = SAGEConv(features, hidden)
#         self.sage2 = SAGEConv(hidden, classes)
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.sage1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.sage2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
#
# class GAT_Net(torch.nn.Module):
#     def __init__(self, features, hidden, classes, heads=1):
#         super(GAT_Net, self).__init__()
#         self.gat1 = GATConv(features, hidden, heads=heads)
#         self.gat2 = GATConv(hidden*heads, classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.gat1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.gat2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GraphSAGE_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
# data = dataset[0]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# acc = int(correct)/ int(data.test_mask.sum())
# print('GraphSAGE',acc)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GAT_Net(dataset.num_node_features, 16, dataset.num_classes, heads=4).to(device)
# data = dataset[0]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
# acc = int(correct)/ int(data.test_mask.sum())
# print('GAT',acc)