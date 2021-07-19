import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import Adam
import pgl


def build_graph():
    # define the number of nodes; we can use number to represent every node
    num_node = 10
    # add edges, we represent all edges as a list of tuple (src, dst)
    edge_list = [(2, 0), (2, 1), (3, 1),(4, 0), (5, 0),
             (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (9, 7)]

    # Each node can be represented by a d-dimensional feature vector, here for simple, the feature vectors are randomly generated.
    d = 16
    feature = np.random.randn(num_node, d).astype("float32")
    # each edge has it own weight
    edge_feature = np.random.randn(len(edge_list), 1).astype("float32")

    print("feature=",feature.shape)
    print("egde_feature=",edge_feature.shape)
    # create a graph
    g = pgl.Graph(edges = edge_list,
                  num_nodes = num_node,
                  node_feat = {'nfeat':feature},
                  edge_feat ={'efeat': edge_feature})

    return g

g = build_graph()
print(type(g))

# print('There are %d nodes in the graph.'%g.num_nodes)
# print('There are %d edges in the graph.'%g.num_edges)
print("node_feature=",g.node_feat)