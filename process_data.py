import os
import pdb
import argparse

import scipy.io
import numpy as np

import networkx as nx

def author_matrix(mat_obj):
    da = np.array(yy['docs_authors'].todense())
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
    yy = scipy.io.loadmat('build/nips_1-17.mat')
    names = np.squeeze(yy['authors_names'])
    RR=author_matrix(yy)
    write_author_matrix_gml('build/nips-full.gml', RR, names)
    return nx.read_gml(os.path.join('build', 'nips-full.gml'))
    
def get_234(net):
    # restricted the network to the 234 most connected individuals
    # Parameters
    # net: nx.Digraph
    if(os.path.exists('build/nips-234.gml')):
        return    
    tmp_list = []
    try:
        for i in net.nodes:
            tmp_list.append([i, net.degree(i, weight='weight')])
    except Exception as e:
        pdb.set_trace()
    tmp_list.sort(key=lambda x:x[1], reverse=True)
    node_list = []
    for i in range(234):
        node_list.append(tmp_list[i][0])
    sub = net.subgraph(node_list)
    nx.write_gml(sub, os.path.join('build', 'nips-234.gml'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, type=bool, nargs='?', const=True, help='whether to enter debug mode')                  
    args = parser.parse_args()
    if(args.debug):
        pdb.set_trace()
    net = process_full_data()
    get_234(net)