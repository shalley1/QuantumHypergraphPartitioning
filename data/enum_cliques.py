import networkx as nx
import numpy as np

import pynauty as pn

def is_asymmetric_pynauty(G: nx.Graph) -> bool:
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    pg = pn.Graph(number_of_vertices=G.number_of_nodes(),
                  adjacency_dict=adj,
                  directed=False)
    return pn.autgrp(pg)[4]

def process(f):
    G = nx.convert_node_labels_to_integers(nx.read_edgelist(f, data=(("weight", float),)))
    clq = list(nx.enumerate_all_cliques(G))
    cliques = list(filter(lambda x: not any(map(lambda y: set(x) < set(y), clq)), clq))
    print([len(x) for x in cliques], len(cliques))
    P = np.zeros((len(G), len(cliques)))
    for (i, e) in enumerate(cliques):
        for v in e:
            P[v, i] = 1 / len(e)
    np.save(f"./{f}_enumerated.npy", P, False)

if __name__ == '__main__':
    process("./Karloff_6_3_1.edgelist")