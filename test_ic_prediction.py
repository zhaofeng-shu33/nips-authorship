import unittest

import networkx as nx

from info_cluster import InfoCluster
from ic_prediction import info_clustering_prediction
from evaluation import evaluate_single

class TestIcPrediction(unittest.TestCase):
    def test_cv_value(self):
        g = nx.Graph() # undirected graph
        g.add_edge(0, 1, weight=1)
        g.add_edge(1, 2, weight=1)
        g.add_edge(0, 2, weight=5)
        ic = InfoCluster(affinity='precomputed') # use precomputed graph structure
        ic.fit(g)
        self.assertAlmostEqual(ic.tree.cv, 2.0)
        self.assertAlmostEqual(ic.tree.get_children()[0].cv, 5.0)
        
    def test_predict_trivial(self):
        g = nx.Graph() # undirected graph
        g.add_edge(0, 1, weight=1)
        ic = info_clustering_prediction()
        ic.fit(g)
        self.assertTrue(ic.predict(0, 1))
        self.assertTrue(ic.predict(1, 0))
    
    def test_predict_with_same_ancestor(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=1)
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=1)
        ic = info_clustering_prediction()
        ic.fit(g, weight_method=None)
        self.assertTrue(ic.predict(0, 3))   
        self.assertTrue(ic.predict(3, 0))   
        self.assertFalse(ic.predict(0, 2))   
        self.assertFalse(ic.predict(2, 0))   

    def test_predict_with_different_ancestor_false(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=1)
        g.add_edge(0, 2, weight=1)
        g.add_edge(1, 2, weight=1)
        g.add_node(3)
        ic = info_clustering_prediction()
        ic.fit(g, weight_method=None)
        self.assertFalse(ic.predict(0, 3))   
        self.assertFalse(ic.predict(3, 0))

    def test_predict_with_different_ancestor_false_2(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=1)
        g.add_edge(2, 3, weight=1)
        ic = info_clustering_prediction()
        ic.fit(g, weight_method=None)
        self.assertFalse(ic.predict(0, 3))   

    def test_predict_with_different_ancestor_true(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=2)
        g.add_edge(1, 2, weight=1.1)
        g.add_edge(2, 3, weight=1)
        g.add_edge(3, 4, weight=1)
        g.add_edge(2, 4, weight=1)
        ic = info_clustering_prediction()
        ic.fit(g, weight_method=None)
        self.assertTrue(ic.predict(0, 2))   
        self.assertTrue(ic.predict(2, 0))   

    def test_evaluation_single(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=1)
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=1)
        ic = info_clustering_prediction()        
        ic.fit(g, weight_method=None)
        test_index_list = [(0, 3)]
        res = evaluate_single(ic, test_index_list, need_fit=False)
        self.assertAlmostEqual(res["tpr"], 1.0)
        self.assertAlmostEqual(res["tnr"], 1.0)
        self.assertAlmostEqual(res["acc"], 1.0)

if __name__ == '__main__':
    unittest.main()