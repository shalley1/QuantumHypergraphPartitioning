from scipy.stats import poisson
import numpy as np
import networkx as nx

def poisson_edges(n, d, o, rng=np.random.default_rng()):
    m = rng.poisson(n * d)
    return [rng.choice(n, size=rng.poisson(o-2)+2 if o > 2 else 2, replace=False) for _ in range(m)]

if __name__ == '__main__':
    path = "./data/Karloff_6_3_1.edgelist"
    K = nx.convert_node_labels_to_integers(nx.read_edgelist(path, data=False))
    B = nx.incidence_matrix(K)
    clq = list(nx.enumerate_all_cliques(K))
    cliques = list(filter(lambda x: not any(map(lambda y: set(x) < set(y), clq)), clq))
    
    p, d, o = 0.0, 0, 0
    for idx in range(1):
        rng = np.random.default_rng(idx)
        edges = list(filter(lambda  _: rng.random() > p,  cliques)) #  + poisson_edges(len(K), d, o, rng)
        Pi = np.zeros((len(K), len(edges)))
        for (i, e) in enumerate(edges):
            for v in e:
                Pi[v, i] = 1 / len(e)
        np.save(f"./data/karloff/numpy_20/6_3_1_{idx}.npy", Pi, False)

