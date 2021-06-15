import argparse
import multiprocessing
from collections import defaultdict
from operator import index

import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm

from walk import RWGraph


class Vocab(object):

    def __init__(self, count, index):
        self.count = count
        self.index = index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/amazon',
                        help='Input dataset path')
    
    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--walk-file', type=str, default=None,
                        help='Input random walks')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')
    
    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')
    
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for generating random walks. Default is 16.')

    return parser.parse_args()

def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict  # 每个节点和它相连接的节点

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()  # 每个type对应到的相连接节点
    all_nodes = list()  # 所有节点的集合
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')  # 去除/n
            if words[0] not in edge_data_by_type:  # edge type涉及到的节点
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))  # nodes去重
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type  # 每个type连接的点边情况


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()  # true样本
    false_edge_data_by_type = dict()  # false样本
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:  # true对应到的节点
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()  # true对应到的type相连接节点
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type

def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

def load_feature_data(f_name):
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic

def generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers):
    if schema is not None:  # schema：节点类型
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []  # 所有游走的list
    for layer_id, layer_name in enumerate(network_data):
        tmp_data = network_data[layer_name]  # 每个type对应到的点边信息
        # start to do the random walk on a layer
        # get_G_from_edges(tmp_data): 每个节点对应到相连接的点
        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)  # RandomWalk Graph
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)  # 生成随机游走的序列; 每个节点游走次数; 游走长度;

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks

def generate_pairs(all_walks, vocab, window_size, num_workers):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generating training pairs for layer', layer_id)
        for walk in tqdm(walks):  # type的游走语料进行循环
            for i in range(len(walk)):  # 每个单词循环
                for j in range(1, skip_window + 1):  # 向前向后的窗口长度
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))  # 向前窗口涉及到的单词
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))  # 向后窗口涉及到的单词
    return pairs  # 所有单词上线文的索引, type

def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    # 随机游走每个单词出现的次数
    for layer_id, walks in enumerate(all_walks):  # 按照type类别
        print('Counting vocab for layer', layer_id)
        for walk in tqdm(walks):
            for word in walk:  # 记录每个单词出现的次数
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))  # 用一个类表示节点的次数和index
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)  # 按出现次数降序排列
    for i, word in enumerate(index2word):  # 节点按出现次数降序更新索引
        vocab[word].index = i
    # 节点统计信息; 节点信息
    return vocab, index2word

def load_walks(walk_file):
    print('Loading walks')
    all_walks = []
    with open(walk_file, 'r') as f:
        for line in f:
            content = line.strip().split()
            layer_id = int(content[0])
            if layer_id >= len(all_walks):
                all_walks.append([])
            all_walks[layer_id].append(content[1:])
    return all_walks

def save_walks(walk_file, all_walks):
    with open(walk_file, 'w') as f:
        for layer_id, walks in enumerate(all_walks):
            print('Saving walks for layer', layer_id)
            for walk in tqdm(walks):
                f.write(' '.join([str(layer_id)] + [str(x) for x in walk]) + '\n')

def generate(network_data, num_walks, walk_length, schema, file_name, window_size, num_workers, walk_file):
    if walk_file is not None:  # 如果没有walk_file，则自己生成walks
        all_walks = load_walks(walk_file)
    else:  # 边的信息; 每个节点游走次数; 游走长度;
        all_walks = generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers)  # 生成随机游走序列
        save_walks(file_name + '/walks.txt', all_walks)
    vocab, index2word = generate_vocab(all_walks)  # 生成节点index; 降序方式;
    train_pairs = generate_pairs(all_walks, vocab, window_size, num_workers)  # (center, context, type)
    # vocab:节点统计信息; index2word:节点; train_pairs:skip-gram训练样本
    return vocab, index2word, train_pairs

def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]  # 每个type涉及到的节点
        for (x, y) in tqdm(g):
            ix = vocab[x].index  # x对应到的索引
            iy = vocab[y].index  # y对应到的索引
            neighbors[ix][r].append(iy)  # 邻居信息
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:  # 节点在这个类别下，如果没有节点和它连接，邻居就是该节点本身
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:  # 如果邻居节点数量小于采样邻居数量，进行重采样
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:  # 如果邻居节点数量大于采样邻居数量，进行邻居大小数量的采样
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors  # 每个节点的邻居采样

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1  # 预测输出的结果

    y_true = np.array(true_list)  # true label
    y_scores = np.array(prediction_list)  # predict proba
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)
