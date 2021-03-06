import numpy as np
import networkx as nx

def train_test_split(net, ratio = 0.2):
    # split the edges into train and test set
    # return (train_net, test_edge_list)
    total_num = len(net.edges)
    test_num = int(total_num * ratio)
    test_index = np.random.choice(total_num, test_num, replace=False)
    test_edge_list = []
    train_net = net.copy()
    edge_list = []
    for i in net.edges:
        edge_list.append(i)
    for i in test_index:
        u, v = edge_list[i]
        test_edge_list.append((int(u), int(v)))
        train_net.remove_edge(u, v)
    return (nx.convert_node_labels_to_integers(train_net), test_edge_list)