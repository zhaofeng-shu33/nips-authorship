'''
'''
import random
import argparse
from datetime import datetime
import pdb
import logging
import os
import json

import numpy as np
import networkx as nx # for manipulating graph data-structure
from ete3 import Tree
from sklearn.model_selection import KFold

from info_cluster import InfoCluster
from utility import construct_subgraph

LOGGING_FILE = 'nips_authorship_%d.log'%os.getpid()
logging.basicConfig(filename=os.path.join('build', LOGGING_FILE), level=logging.INFO, format='%(asctime)s %(message)s')

color_list = ['red', 'orange', 'green', 'purple']
shape_list = ['sphere', 'circle', 'sphere', 'sphere']

            
def plot_clustering_tree(tree, alg_name, cutting=0):
    '''if cutting=True, merge the n nodes at leaf nodes with the same parent.
    '''
    if(cutting):
        tree_inner = tree.copy()
        cnt = 0
        delete_tree_node_list = []
        for _n in tree_inner:
            try:
                _n.category
            except AttributeError:
                _n.add_features(category=cnt)
                for i in _n.get_sisters():
                    if not(i.is_leaf()):
                        continue
                    try:
                        i.category
                    except AttributeError:
                        i.add_features(category=cnt)
                        delete_tree_node_list.append(i)
                cnt += 1
        for _n in delete_tree_node_list:
            _n.delete()
        # rename the tree node
        tree_inner = Tree(tree_inner.write(features=[]))
    else: 
        tree_inner = tree

        
    ts = TreeStyle()
    ts.rotation = 90
    ts.show_scale = False
    # for _n in tree_inner:
        # nstyle = NodeStyle()
        # nstyle['fgcolor'] = color_list[int(_n.macro)]
        # nstyle['shape'] = shape_list[int(_n.micro)]
        # _n.set_style(nstyle)
    time_str = datetime.now().strftime('%Y-%m-%d-')    
    tree_inner.render(os.path.join('build', time_str + 'tree.pdf'.replace('.pdf', '_' + alg_name + '.pdf')), tree_style=ts)
    


    
def evaluate(num_times, alg, z_in_1, z_in_2, z_o):
    '''
        num_times: int
        alg: algorithm class
        z_in_1: inter-micro-community node average degree     
        z_in_2: intra-micro-community node average degree
        z_o: intra-macro-community node average degree
        
        the evaluated alg is a class, and should provide fit method , which operates on similarity matrix
        and get_category(i) method, where i is the specified category.
    '''
    report = {'norm_rf' : 0,
             }
    assert(z_in_1 > z_in_2 and z_in_2 > z_o)
    logging.info('eval ' + str(type(alg)) + ' num_times=%d, z_in_1=%f,z_in_2=%f, z_o=%f'%(num_times, z_in_1, z_in_2, z_o))
    for i in range(num_times):
        G = construct(z_in_1, z_in_2, z_o)
        norm_rf = evaluate_single(alg, G)
        logging.info('round {0}: with norm_rf={1}'.format(i, norm_rf))
        report['norm_rf'] += norm_rf
    report['norm_rf'] /= num_times
    report.update({
                'num_times': num_times,
                'z_in_1': z_in_1,
                'z_in_2': z_in_2,
                'z_o': z_o})
    return report
    

def train_test_split(alg, net):
    # split the edges into train and test set
    # return (train_set, test_set)
    # train_set or test_set is the collection of edges
    kf = KFold(n_splits=10, random_state=None, shuffle=False)    
    for train_index_list, test_index_list in kf.split(net.edges):
        sub = construct_subgraph(train_index_list)
        res = evaluate_single(alg, sub, test_index_list)
        # average over different cv
def write_gml_wrapper(G, filename, ignore_attr=False):
    if(ignore_attr):
        _G = nx.Graph()
        for node in G.nodes():
            _G.add_node(node)
        for edge in G.edges():
            i,j = edge
            _G.add_edge(i,j)
            
        # remove the attribute of _G
    else:
        _G = G.copy()
        info_clustering_add_weight(_G)
    nx.write_gml(_G, filename)
        

def save_tree_txt(tree, alg_name):
    tree_txt = tree.write()
    time_str = datetime.now().strftime('%Y-%m-%d-')    
    
    write_file_name = os.path.join('build', time_str + '_' + alg_name + '_tree.nw')
    with open(write_file_name, 'w') as f:
        f.write(tree_txt)
    
class InfoClusterWrapper(InfoCluster):
    def __init__(self):
        super().__init__(affinity='precomputed')
    def fit(self, _G, weight_method='triangle-power'):
        G = _G.copy()
        if(weight_method=='triangle-power'):            
            info_clustering_add_weight(G)
        try:
            super().fit(G, use_psp_i=True)
        except RuntimeError as e:
            print(e)
            # dump the graph
            print('internal error of the pdt algorithm, graph dumped to build/graph_dump.gml')
            nx.write_gml(_G, os.path.join('build', 'graph_dump.gml'))
            
if __name__ == '__main__':
    method_chocies = ['info-clustering', 'gn', 'bhcd', 'all']
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_graph', default=0, type=int, help='whether to save gml file, =0 not save(default), =1 save complete, =2 save without attribute')
    parser.add_argument('--load_graph', help='use gml file to initialize the graph')     
    parser.add_argument('--save_tree', default=0, type=int, help='whether to save the clustering tree file after clustering, =0 not save, =1 save original(pdf), =2 save simplified(pdf), =3 save ete format txt')     
    parser.add_argument('--alg', default='all', choices=method_chocies, help='which algorithm to run', nargs='+')
    parser.add_argument('--weight', default='triangle-power', help='for info-clustering method, the edge weight shold be used. This parameters'
        ' specifies how to modify the edge weight.')    
    parser.add_argument('--debug', default=False, type=bool, nargs='?', const=True, help='whether to enter debug mode')                  
    parser.add_argument('--evaluate', default=2, type=int, help='when evaluate=1, evaluate the method using norm rf = 2; when evaluate=2, compare with ground truth; evaluate=0, no evaluation.')                      
    args = parser.parse_args()
    method_chocies.pop()
    if(args.debug):
        pdb.set_trace()
    if(args.load_graph):
        G = nx.read_gml(os.path.join('build', args.load_graph))
    else:
        G = construct(args.z_in_1, args.z_in_2, z_o)    
    if(args.plot_graph):
        graph_plot(G)
    if(args.save_graph):
        write_gml_wrapper(G, 'build/tuning.gml', args.save_graph-1)
    methods = []
    if(args.alg.count('all')>0):
        args.alg = method_chocies
    if(args.alg.count('info-clustering')>0):
        methods.append(InfoClusterWrapper())
    if(args.alg.count('gn')>0):
        methods.append(GN())
    if(args.alg.count('bhcd')>0):
        methods.append(BHCD(restart=bhcd_parameter.restart, 
            gamma=bhcd_parameter.gamma, _lambda=bhcd_parameter._lambda, delta=bhcd_parameter.delta))
    if(len(methods)==0):
        raise ValueError('unknown algorithm')
    
    if(args.evaluate == 2):
        print('logging to', LOGGING_FILE)
        for method in methods:
            report = evaluate(args.evaluate, method, args.z_in_1, args.z_in_2, z_o)
            logging.info('final report' + json.dumps(report))
    elif(args.evaluate == 1):
        for i, method in enumerate(methods):
            alg_name = args.alg[i]
            print('running ' + alg_name)
            dis = evaluate_single(method, G)            
            print('tree distance is', dis)           
    else:
        for method in methods:
            method.fit(G)
    if(args.save_tree):
        for i, method in enumerate(methods):
            alg_name = args.alg[i]
            if(args.save_tree == 3):
                save_tree_txt(method.tree, alg_name)
            else:
                plot_clustering_tree(method.tree, alg_name, args.save_tree - 1)
