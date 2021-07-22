import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd

df = pd.DataFrame()
df['source'] = [0,1]
df['target'] = [1,2]
g = nx.from_pandas_edgelist(df,create_using=nx.DiGraph())
print(list(g.neighbors(0)))