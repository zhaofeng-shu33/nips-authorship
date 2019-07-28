import networkx as nx

def construct_subgraph(index_list):
    g = nx.Graph()
    for i,j in index_list:
        g.add_edge(i, j)