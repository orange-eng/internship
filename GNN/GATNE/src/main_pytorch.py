import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter

from utils import *


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size  # 迭代次数

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []  # src, dst, type, neigh
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])  # src的邻居节点
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class GATNEModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    ):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes  #   节点数量
        self.embedding_size = embedding_size  # 每个节点输出的embedding_size
        self.embedding_u_size = embedding_u_size  # 节点作为邻居初始化size
        self.edge_type_count = edge_type_count  # 类别数量
        self.dim_a = dim_a  # 中间隐层特征数量

        self.features = None
        if features is not None:  # GATNE-I
            self.features = features
            feature_dim = self.features.shape[-1]
            self.embed_trans = Parameter(torch.FloatTensor(feature_dim, embedding_size))  # [142, 200]; bi-base embedding
            self.u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))  # [2, 142, 10]; 初始化ui
        else:  # 初始化 base embedding GATNE-T
            self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))  # [511, 200]
            self.node_type_embeddings = Parameter(  # 初始化 edge embedding
                torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
            )  # [511, 2, 10]
        self.trans_weights = Parameter(  # [2, 10, 200]; 定义Mr矩阵
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(  # [2, 10, 20]  计算attention使用
            torch.FloatTensor(edge_type_count, embedding_u_size, dim_a)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))  # [2, 20, 1]

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            self.embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
            self.u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        else:
            self.node_embeddings.data.uniform_(-1.0, 1.0)
            self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        if self.features is None:
            node_embed = self.node_embeddings[train_inputs]  # 每个节点对应的mebedding
            node_embed_neighbors = self.node_type_embeddings[node_neigh]  # 每个节点对应的neighbors
        else:  # self.features:节点特征; self.embed_trans
            node_embed = torch.mm(self.features[train_inputs], self.embed_trans)  # [64, 200]
            node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)  # 生成ui; [64, 2, 10, 142]*[2, 142, 10];
        node_embed_tmp = torch.cat(  # [64, 2, 10, 10]; 聚合每个类别周围邻居信息
            [
                node_embed_neighbors[:, i, :, i, :].unsqueeze(1)  # [64, 1, 10, 10]
                for i in range(self.edge_type_count)
            ],
            dim=1,
        )
        node_type_embed = torch.sum(node_embed_tmp, dim=2)  # Ui; 对邻居信息求和; [64, 2, 10]

        trans_w = self.trans_weights[train_types]  # [64, 10, 200]
        trans_w_s1 = self.trans_weights_s1[train_types]  # [64, 10, 20]
        trans_w_s2 = self.trans_weights_s2[train_types]  # [64, 20, 1]

        attention = F.softmax(  # [64, 1, 2]
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)  # [64, 1, 2] * [64, 2, 10] 对node_type_embed做attention求和
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)  # [64, 200] + [64, 1, 10] * [64, 10, 200] => [64, 200]

        last_node_embed = F.normalize(node_embed, dim=1)  # dim=1, L2-norm; (last_node_embed*last_node_embed).sum(axis=1)

        return last_node_embed


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))  # [511, 200]; Cj
        self.sample_weights = F.normalize(  # [511]; 对节点进行初始化
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(  # torch.mul:对应位置相乘
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))  # sigmoid([64]); input_embeddings*labels_embeddings
        )
        negs = torch.multinomial(  # 抽样函数，self.sample_weights的权重抽样
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])  # Cj; 所有值 * -1
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1  # [64, 5, 1]
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


def train_model(network_data, feature_dic):
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)
    # 生成随机游走训练序列和训练语料
    edge_types = list(network_data.keys())  # 边的类别

    num_nodes = len(index2word)  # 节点数量
    edge_type_count = len(edge_types)  # 类别数量
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions  # base embedding_size; embedding_size;
    embedding_u_size = args.edge_dim  # edge_embedding_size
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim  # 计算attention的中间变量维度
    att_head = 1
    neighbor_samples = args.neighbor_samples  # 邻居采样数量

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 每个类别节点的邻居节点; 在计算ui时使用;
    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    features = None
    if feature_dic is not None:  # GATNE-I
        feature_dim = len(list(feature_dic.values())[0])  # 特征长度
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)  # feature array
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        features = torch.FloatTensor(features).to(device)  # 特征矩阵
    # 建立GATNE model
    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4
    )

    best_score = 0
    test_score = (0.0, 0.0, 0.0)
    patience = 0
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)

        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            optimizer.zero_grad()  # center, context, types, neigh
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device),)  # 节点的embeddings
            loss = nsloss(data[0].to(device), embs, data[1].to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

            '''  调试使用  '''
            if i==0:
                break

        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))  # 每个类别下节点的embedding;
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)  # 节点的多个类别，求每个类别下的embedding
            train_types = torch.tensor(list(range(edge_type_count))).to(device)
            node_neigh = torch.tensor(  # 节点在每个类别下的neighbors
                [neighbors[i] for _ in range(edge_type_count)]
            ).to(device)
            node_emb = model(train_inputs, train_types, node_neigh)  # [node1, node1]; [type1, type2]; [node1_neigh, node1_neigh]
            for j in range(edge_type_count):  # 每个节点在各个类别下的embedding
                final_model[edge_types[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []
        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        print("valid auc:", np.mean(valid_aucs))
        print("valid pr:", np.mean(valid_prs))
        print("valid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)

        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            best_score = cur_score
            test_score = (average_auc, average_f1, average_pr)
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping")
                break
    return test_score


if __name__ == "__main__":
    args = parse_args()
    args.input = '../data/example'
    args.features = '../data/example/feature.txt'
    file_name = args.input
    print(args)
    if args.features is not None:  # 每个节点对应到的特征; GATNE-T;
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None

    training_data_by_type = load_training_data(file_name + "/train.txt")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/test.txt"
    )

    average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic)

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
