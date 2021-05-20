import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd

df = pd.DataFrame()
df['source'] = [1,1,1,2,2,3,3,4,4,5,5,5]    #表示起始点
df['targets'] = [2,4,5,3,1,2,5,1,5,1,3,4]   #表示终止点
df['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1]   #从起始点到终止点连线的权重
G = nx.from_pandas_edgelist(df,source='source',target='targets',edge_attr='weights')

#g = nx.from_pandas_edgelist(df,create_using=nx.DiGraph())
#print(list(g.neighbors(0)))
print(nx.degree(G))
print(list(nx.connected_components(G)))
print(nx.degree_centrality(G))
