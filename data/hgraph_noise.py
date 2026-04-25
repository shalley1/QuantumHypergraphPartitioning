from scipy.stats import poisson
import numpy as np
import networkx as nx
from collections import Counter
from pathlib import Path


def poisson_edges(n, d, o, rng=np.random.default_rng()):
    m = rng.poisson(n * d)
    return [rng.choice(n, size=rng.poisson(o-2)+2 if o > 2 else 2, replace=False) for _ in range(m)]

def poisson_hgraph(m, n, kappa, rng=np.random.default_rng()):
    edges = Counter(
        frozenset(rng.choice(n, size=rng.poisson(kappa-2)+2 if kappa > 2 else 2, replace=False))
        for _ in range(m)
    )
    P = np.zeros((n, len(edges)))
    w = np.zeros(len(edges))
    for (i, (key, val)) in enumerate(edges.items()):
        w[i] = val
        for v in key:
            P[v, i] = 1/len(key)
    return P, w


if __name__ == '__main__':
    # path = "./data/Karloff_6_3_1.edgelist"
    # K = nx.convert_node_labels_to_integers(nx.read_edgelist(path, data=False))
    # B = nx.incidence_matrix(K)
    # clq = list(nx.enumerate_all_cliques(K))
    # cliques = list(filter(lambda x: not any(map(lambda y: set(x) < set(y), clq)), clq))
    ngraphs = 10
    kappas = [3, 4, 5]
    m, n = 16, 12
    for kappa in kappas:
        root = Path(f"data/poisson_{m}_{n}_{kappa}")
        (root / "P").mkdir(parents=True, exist_ok=True)
        (root / "w").mkdir(parents=True, exist_ok=True)
        for idx in range(1, ngraphs+1):
            rng = np.random.default_rng(idx)
            P, w = poisson_hgraph(m, n, kappa, rng)
            sidx = str(idx).zfill(len(str(ngraphs)))
            np.save(root / f"P/{sidx}.npy", P, False)
            np.save(root / f"w/{sidx}.npy", w, False)

