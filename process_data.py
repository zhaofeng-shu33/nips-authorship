import os

import scipy.io
import numpy as np



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
        return
    yy = scipy.io.loadmat('build/nips_1-17.mat')
    names = np.squeeze(yy['authors_names'])
    RR=author_matrix(yy)
    write_author_matrix_gml('build/nips-full.gml', RR, names)

if __name__ == '__main__':
    process_full_data()