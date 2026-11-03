from lp import fair_min_bisection
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import glob
import pandas as pd

def get_edge_list(P):
    edges=[]
    for idx in range(P.shape[1]):
        #indices that are not 0
        edge = tuple(np.flatnonzero(P[:,idx]))
        edges.append(edge)
    return edges

poisson_types = [
    "poisson_16_12_3",
    "poisson_16_12_4",
    "poisson_16_12_5"
]

poisson_results = []

for graph_type in poisson_types:

    p_folder = f"../data/{graph_type}/P"
    w_folder = f"../data/{graph_type}/w"

    p_files = sorted(glob.glob(f"{p_folder}/*.npy"))

    for p_file in p_files:
        instance = os.path.splitext(os.path.basename(p_file))[0]

        # Load stochastic incidence matrix
        P = np.load(p_file)
        n = P.shape[0]
        edges = get_edge_list(P)

        # Load hyperedge weights
        w_file = os.path.join(w_folder, f"{instance}.npy")
        weights = np.load(w_file)
        weights = np.asarray(weights).reshape(-1)

        assert len(weights) == len(edges), (
            f"{graph_type}/{instance}: "
            f"{len(weights)} weights different than "
            f"{len(edges)} edges"
        )

        # Solve weighted balanced fair cut cover LP
        partitions, probs, cut_probs, optimal_t = (
            fair_min_bisection(edges, weights, n)
        )

        # Weighted coverage of each hyperedge
        weighted_coverage = weights * cut_probs

        # Number of partitions assigned positive probability
        num_support = int(np.sum(probs > 1e-8))

        # Minimum unweighted cut probability
        max_cut_probability = float(np.max(cut_probs))

        poisson_results.append({
            "dataset": graph_type,
            "instance": instance,
            "n_vertices": n,
            "n_edges": len(edges),
            "optimal_t": optimal_t,
            "max_cut_probability": max_cut_probability,
            "num_support_partitions": num_support,
            "cut_probabilities": cut_probs,
            "weighted_coverages": weighted_coverage.tolist()
        })

        print(
            graph_type,
            instance,
            "| optimal t =", round(optimal_t, 6),
            "| max cut probability =", round(max_cut_probability, 6),
            "| support =", num_support
        )

# Convert results to DataFrame
poisson_df = pd.DataFrame(poisson_results)
poisson_df.to_csv("poisson_min_fair_LP.csv",index=False)
