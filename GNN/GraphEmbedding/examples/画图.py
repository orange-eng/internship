import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

# load the karate club graph
G = nx.karate_club_graph()

#first compute the best partition
partition = community_louvain.best_partition(G)

dictionary = dict(partition)
print(dictionary)

pr = nx.pagerank(G)
print(pr)

# draw the graph
pos = nx.spring_layout(G)
print(pos)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))

plt.show()



'''
2.
'''

nx.draw(G, node_size=50,  font_size=10, font_color="blue", font_weight="bold")
plt.show()
