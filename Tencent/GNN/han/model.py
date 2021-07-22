import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        # input:[Node, metapath, in_size]; output:[None, metapath, 1]; 所有节点在每个meta-path上的重要性值
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)    # 每个节点在metapath维度的均值; mean(0): 每个meta-path上的均值(/|V|); (MetaPath, 1)
        beta = torch.softmax(w, dim=0)       # 归一化          # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) #  拓展到N个节点上的metapath的值   (N, M, 1)

        return (beta * z).sum(1)     #  (beta * z)=>所有节点，在metapath上的attention值;    (beta * z).sum(1)=>节点最终的值      (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):  # meta-path Layers; 两个meta-path的维度是一致的
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)  # 语义attention; out-size*layers
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []  # 语义级别的embeddings
        # 每个meta-path下对应到的节点级别的attention layer
        for i, g in enumerate(gs):  # 每个meta-path的图信息，求节点的attention. 两个GAT; gat_layers[i](g, h) g:图; h:features => [3025, 8, 8] => [3025, 64]
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))  # 两个GAT; gat_layers[i](g, h) 每个metapath对应一个GAT
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)   # (N, M, D * K) 每个节点对应到metapath下的每个节点的embedding值（Node Attention）
        # 聚合meta-path下，每个节点最终的输出值
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout)) # meta-path数量 + semantic_attention
        for l in range(1, len(num_heads)): # 多层多头，目前是没有
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)  # hidden*heads, classes; HAN->classes

    def forward(self, g, h):
        for gnn in self.layers:  # GAT-GAT 节点级别的GAT; semantic_attention语义级别attention;
            h = gnn(g, h)  # HANLayer
            # 输出的是：节点, meta-path数量, embedding; Returns:节点HAN后输出的embedding
        return self.predict(h)  # HAN输出节点embedding后接一个Linear层
