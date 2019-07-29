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
try:
    from ete3 import TreeStyle, NodeStyle
except ImportError:
    pass
from sklearn.model_selection import KFold

from ic_prediction import info_clustering_prediction
from utility import train_test_split
from evaluation import evaluate_single

LOGGING_FILE = 'nips_authorship.log'
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
    time_str = datetime.now().strftime('%Y-%m-%d-')    
    tree_inner.render(os.path.join('build', time_str + alg_name + '.pdf'), tree_style=ts)
    

def evaluate_single_wrapper(alg, G):
    new_g, test_edge_list = train_test_split(G)
    return evaluate_single(alg, test_edge_list, new_g)

def evaluate(num_times, alg, G):
    '''
        num_times: int
        alg: algorithm class
        z_in_1: inter-micro-community node average degree     
        z_in_2: intra-micro-community node average degree
        z_o: intra-macro-community node average degree
        
        the evaluated alg is a class, and should provide fit method , which operates on similarity matrix
        and get_category(i) method, where i is the specified category.
    '''
    report = {
        "tpr": 0,
        "tnr": 0,
        "acc": 0
    }
    
    logging.info('eval ' + str(type(alg)) + ' num_times=%d'%(num_times))
    for i in range(num_times):
        res = evaluate_single_wrapper(alg, G)
        logging.info('round {0}: with res = {1}'.format(i, res))
        for k in report:
            report[k] += res[k]
    for k in report:
        report[k] /= num_times
    return report
        
def save_tree_txt(tree, alg_name):
    tree_txt = tree.write()
    time_str = datetime.now().strftime('%Y-%m-%d-')    
    
    write_file_name = os.path.join('build', time_str + '_' + alg_name + '_tree.nw')
    with open(write_file_name, 'w') as f:
        f.write(tree_txt)

            
if __name__ == '__main__':
    method_chocies = ['info-clustering', 'bhcd', 'all']
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_graph', default=0, type=int, help='whether to save gv file')
    parser.add_argument('--load_graph', help='use custom gml file to initialize the graph')     
    parser.add_argument('--save_tree', default=0, type=int, help='whether to save the clustering tree file after clustering, =0 not save, =1 save original(pdf), =2 save simplified(pdf), =3 save ete format txt')     
    parser.add_argument('--alg', default='all', choices=method_chocies, help='which algorithm to run', nargs='+')
    parser.add_argument('--weight', default='triangle-power', help='for info-clustering method, the edge weight shold be used. This parameters'
        ' specifies how to modify the edge weight.')    
    parser.add_argument('--debug', default=False, type=bool, nargs='?', const=True, help='whether to enter debug mode')                  
    parser.add_argument('--evaluate', default=1, type=int, help='when evaluate=1, evaluate the method once; when evaluate=2, iterate given times; evaluate=0, no evaluation.')
    parser.add_argument('--num_times', default=10, type=int, help='the number of times of evaluation')                      
    args = parser.parse_args()
    method_chocies.pop()
    if(args.debug):
        pdb.set_trace()
    if(args.load_graph):
        G = nx.read_gml(os.path.join('build', args.load_graph))
    else:
        G = nx.read_gml(os.path.join('build', 'nips-234.gml'))    
    if(args.save_graph):
        raise NotImplementedError("")
    methods = []
    if(args.alg.count('all')>0):
        args.alg = method_chocies
    if(args.alg.count('info-clustering')>0):
        methods.append(info_clustering_prediction())
    if(args.alg.count('bhcd')>0):
        methods.append(BHCD(restart=bhcd_parameter.restart, 
            gamma=bhcd_parameter.gamma, _lambda=bhcd_parameter._lambda, delta=bhcd_parameter.delta))
    if(len(methods)==0):
        raise ValueError('unknown algorithm')
    
    if(args.evaluate == 2):
        print('logging to', LOGGING_FILE)
        for method in methods:
            report = evaluate(args.num_times, method, G)
            logging.info('final report' + json.dumps(report))
    elif(args.evaluate == 1):
        for i, method in enumerate(methods):
            alg_name = args.alg[i]
            print('running ' + alg_name)
            res = evaluate_single_wrapper(method, G)            
            print('evaluation result for ', alg_name, res)           
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
