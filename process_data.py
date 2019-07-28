import os
import pdb
import argparse
from datetime import datetime

import scipy.io
import numpy as np

import networkx as nx
import graphviz

def author_matrix(mat_obj):
    da = np.array(mat_obj['docs_authors'].todense())
    num_docs, num_authors = da.shape
    adj = np.zeros((num_authors, num_authors))

    docs, authors = np.where(da)
    docs = set(docs)
    for doc in docs:
        authors = np.where(da[doc,:])[0]
        for ii, author in enumerate(authors):
            for other in authors[ii+1:]:
                adj[author, other] = 1
                adj[other, author] = 1
    return adj

def write_author_matrix_gml(fname, RR, names):
    with open(fname, 'w') as out:
        out.write('graph [\n\tsparse 0\n')
        out.write('directed 1\n')
        for x in range(len(names)):
            if sum(RR[x,:]) > 0:
                out.write('\tnode [ id %d label "%s" ]\n' % (x, names[x][0]))
        xs,ys = np.where(RR)
        for xx, yy in zip(xs, ys):
            out.write('\tedge [ source %d target %d weight 1 ]\n' % (
                    xx, yy
                ))
        out.write(']\n')

def process_full_data():
    if(os.path.exists('build/nips-full.gml')):
        return nx.read_gml(os.path.join('build', 'nips-full.gml'))
    if not(os.path.exists('build/nips_1-17.mat')):
        raise FileNotFoundError('You need to download nips_1-17.mat first (See README for detail)')
    yy = scipy.io.loadmat('build/nips_1-17.mat')
    names = np.squeeze(yy['authors_names'])
    RR=author_matrix(yy)
    write_author_matrix_gml('build/nips-full.gml', RR, names)
    return nx.read_gml(os.path.join('build', 'nips-full.gml'))
    
def get_234(net, overwrite):
    # restricted the network to the 234 most connected individuals
    # Parameters
    # net: nx.Digraph
    if(not overwrite and os.path.exists('build/nips-234.gml')):
        return    
    node_list = ['1']
    sub = net
    max_iteration = 10
    iter_cnt = 0
    while(len(node_list)>0 and iter_cnt < max_iteration):
        node_list = []
        for i in sub.nodes:
            if(sub.degree(i) > 8):
                node_list.append(i)
        sub = sub.subgraph(node_list)
        iter_cnt += 1
    tmp_list = []
    for i in sub.nodes:
        tmp_list.append([i, sub.degree(i, weight='weight')])
    tmp_list.sort(key=lambda x:x[1], reverse=True)
    node_list = []
    for i in range(234):
        node_list.append(tmp_list[i][0])
    sub = sub.subgraph(node_list)    
    # ensure the node name are integer    
    nx.write_gml(nx.convert_node_labels_to_integers(sub).to_undirected(), os.path.join('build', 'nips-234.gml'))

def graph_plot(G):
    '''
    generate the plot file which is the input of graphviz.
    G: networkx graph object
    '''
    time_str = datetime.now().strftime('%Y-%m-%d')
    g = graphviz.Graph(filename='nips-234-%s.gv'%time_str, engine='neato') # g is used for plotting
    for i in G.nodes(data=True):
        g.node(str(i[0]), shape='point')
    for e in nx.edges(G):
        i,j = e
        g.edge(str(i), str(j), penwidth="0.3")    
    g.save(directory='build')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, type=bool, nargs='?', const=True, help='whether to enter debug mode')                  
    parser.add_argument('--overwrite', help='whether to overwrite the nips-234.gml file', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--command', default="generate_data", choices=["generate_data", "plot_graph"])                      
    args = parser.parse_args()
    if(args.debug):
        pdb.set_trace()
    if(args.command == 'generate_data'):
        net = process_full_data()    
        get_234(net, args.overwrite)
    elif(args.command == 'plot_graph'):
        if not(os.path.exists('build/nips-234.gml')):
            raise FileNotFoundError('You need to invoke generate_data command first')
        G = nx.read_gml(os.path.join('build', 'nips-234.gml'))
        graph_plot(G)