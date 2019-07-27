# empirical prediction
import numpy as np
import networkx as nx

from info_cluster import InfoCluster

def info_clustering_add_weight(G):
    # G is modified within this function
    # for each edge, the weight equals the number of triangles + beta(default to 1)    
    beta = 1
    for e in G.edges():
        i, j = e
        G[i][j]['weight'] = beta
        for n in G.nodes():
            if(G[i].get(n) is not None and G[j].get(n) is not None):
                G[i][j]['weight'] += 1
        G[i][j]['weight'] = G[i][j]['weight']
        
class info_clustering_prediction(InfoCluster):
    '''
       prediction enhancement of info-clustering
       used to predict edge existance
    '''
    def __init__(self):
        super().__init__(affinity='precomputed')
        
    def fit(self, _G, weight_method='triangle-power'):
        if not(type(_G) is nx.Graph):
            raise ValueError("input graph should be an instance of networkx.Graph")
        self.G = _G.copy()
        if(weight_method=='triangle-power'):            
            info_clustering_add_weight(self.G)
        super().fit(self.G, use_psp_i=True)
        # self.tree is available
    
    def get_weight(self, node_i, node_j):
        w = 0
        for i in node_i.get_leaves():
            for j in node_j.get_leaves():
                i_index = int(i.name)
                j_index = int(j.name)
                if(j_index > i_index):
                    continue
                w += self.G[i_index][j_index]['weight']
        return w
        
    def predict_with_same_ancestor(self, tree_node, node_index_i, node_index_j, weight_added):
        # rerun the info-clustering algorithm, if tree_node has many children, the speed is very slow
        child_node_list = tree_node.get_children()
        n_nodes = len(child_node_list)
        affinity_matrix = np.zeros([n_nodes, n_nodes])
        for ii in range(n_nodes):
            for jj in range(ii+1, n_nodes):
                affinity_matrix[ii, jj] = self.get_weight(child_node_list[ii], child_node_list[jj])
        if(node_index_i < node_index_j):           
            affinity_matrix[node_index_i, node_index_j] = weight_added
        else:           
            affinity_matrix[node_index_j, node_index_i] = weight_added
        new_ic = InfoCluster(affinity='precomputed')
        new_ic.fit(affinity_matrix, use_psp_i=True)            
        is_solution_trivial = len(new_ic.critical_values) > 1
        return is_solution_trivial
        
    def predict_with_different_ancestor(self, parent_node, foreign_node_index, weight_added):
        # get the critical value associated with the parent_node
        gamma_N = parent_node.cv
        # get all the leaf node of parent_node
        w_sum = weight_added
        for i in parent_node.get_leaves():
            i_index = int(i.name)
            if(self.G.has_edge(i_index, foreign_node_index)):
                w_sum += self.G[i_index][foreign_node_index]['weight']
        return w_sum > gamma_N
        
    def predict(self, node_index_i, node_index_j, weight_added = 1):
        if not(type(node_index_i) is int and type(node_index_j) is int):
            raise ValueError("two index should be int typed")
        if not(node_index_i >= 0 and node_index_i < len(self.G) and node_index_j >=0 and node_index_j < len(self.G)):
            raise IndexError("index out of range")
        if(self.G.has_edge(node_index_i, node_index_j)):
            return True
        node_i = self.tree.search_nodes(name = str(node_index_i))[0]
        node_j = self.tree.search_nodes(name = str(node_index_j))[0]
        node_i_parent = node_i.get_ancestors()[0]
        node_j_parent = node_j.get_ancestors()[0]
        if(node_i_parent == node_j_parent):
            return self.predict_with_same_ancestor(node_i_parent, node_index_i, node_index_j, weight_added)
        else:
            j_join_i = self.predict_with_different_ancestor(node_i_parent, node_index_j, weight_added) 
            if(j_join_i):
                return True
            i_join_j = self.predict_with_different_ancestor(node_j_parent, node_index_i, weight_added)
            return i_join_j
            
