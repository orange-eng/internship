import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv #内置的GCNlayer
import dgl
import matplotlib.pyplot as plt
import random
import time
import tqdm
import sys
import os


def construct_graph():
    file_user = './data/user_features.csv'
    file_item = './data/item_features.csv'
    file_edge = './data/JData_Action_201602.csv'
    f_user = pd.read_csv(file_user)
    f_item = pd.read_csv(file_item)
    f_edge = pd.read_csv(file_edge)

    #
    f_edge = f_edge.sample(10000)

    users = set()
    items = set()
    for index, row in f_edge.iterrows():
        users.add(row['user_id'])
        items.add(row['sku_id'])

    user_ids_index_map = {x: i for i, x in enumerate(users)}  # user编号
    item_ids_index_map = {x: i for i, x in enumerate(items)}  # item编号
    user_index_id_map = {i: x for i, x in enumerate(users)}  # index:user
    item_index_id_map = {i: x for i, x in enumerate(items)}  # index:item

    user_item_src = []
    user_item_dst = []
    for index, row in f_edge.iterrows():
        user_item_src.append(user_ids_index_map.get(row['user_id']))  # 获取user的编号
        user_item_dst.append(item_ids_index_map.get(row['sku_id']))  # 获取item编号

    # 构图; 异构图的编号
    '''
    ui = dgl.bipartite((user_item_src, user_item_dst), 'user', 'ui', 'item')  # 构建异构图; bipartite
    iu = dgl.bipartite((user_item_dst, user_item_src), 'item', 'iu', 'user')
    
    hg = dgl.hetero_from_relations([ui, iu])
    '''

    data_dict = {('user', 'item', 'user'): (torch.tensor(user_item_src), torch.tensor(user_item_dst))}
    hg = dgl.heterograph(data_dict)
    return hg, user_index_id_map, item_index_id_map


def parse_trace(trace, user_index_id_map, item_index_id_map):
    s = []
    for index in range(trace.size):
        if index % 2 == 0:
            s.append(user_index_id_map[trace[index]])
        else:
            s.append(item_index_id_map[trace[index]])
    return ','.join(s)

def main():
    hg, user_index_id_map, item_index_id_map = construct_graph()
    meta_path = ['ui','iu','ui','iu','ui','iu']
    num_walks_per_node = 1
    f = open("./output/output_path.txt", "w")
    for user_idx in tqdm.trange(hg.number_of_nodes('user')): #以user开头的metapath
        traces = dgl.contrib.sampling.metapath_random_walk(
            hg=hg, etypes=meta_path, seeds=[user_idx,], num_traces=num_walks_per_node)

        dgl.sampling.random_walk

        tr = traces[0][0].numpy()
        tr = np.insert(tr,0,user_idx)
        res = parse_trace(tr, user_index_id_map, item_index_id_map)
        f.write(res+'\n')
    f.close()


if __name__=='__main__':
    main()