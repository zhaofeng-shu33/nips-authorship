
def evaluate_single(alg, test_edge_index_list, G=None, need_fit=True):
    # first fit if needed
    if(need_fit):
        alg.fit(G)    
    if(G is None):
        G = alg.G
    # then predict
    # get tpr
    tp = 0
    for i, j in test_edge_index_list:
        tp += alg.predict(i, j)
    total_positive_sample = len(test_edge_index_list)
    tpr = tp / total_positive_sample
    # get tnr
    tn = 0
    total_negative_sample = 0
    for i in range(len(G)):
        for j in range(i+1, len(G)):
            if(G.has_edge(i, j) or test_edge_index_list.count((i,j)) > 0):
                continue
            tn += (1 - alg.predict(i, j))
            total_negative_sample += 1
    # pack the result
    tnr = tn / total_negative_sample
    acc = (tp + tn) / (total_positive_sample + total_negative_sample)
    res = {
        "tpr": tpr,
        "tnr": tnr,
        "acc": acc
    }
    return res