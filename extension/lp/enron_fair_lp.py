
from lp import fair_min_bisection,fair_max_bisection
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
# folder = "../../data/congress-bills/numpy_20"
folder = "/Users/stevehalley/Code/Github/QuantumHypergraphPartitioning/data/email-Enron/numpy_15"
files = sorted(glob.glob(f"{folder}/*.npy"))

results = []
for file in files:

    P = np.load(file)
    instance = os.path.splitext(os.path.basename(file))[0]

    n = P.shape[0]
    edges = get_edge_list(P)
    weights = np.ones(len(edges))

     # Solve weighted balanced fair cut cover LP
    try:
        partitions, probs, cut_probs, optimal_t = (fair_min_bisection(edges, weights, n))
    except:
        print(f"Gurobi limitation exceeded on {instance}")
        continue

    # Weighted coverage of each hyperedge
    weighted_coverage = weights * cut_probs

    # Number of partitions assigned positive probability
    num_support = int(np.sum(probs > 1e-8))

    # Minimum unweighted cut probability
    max_cut_probability = float(np.max(cut_probs))

    results.append({
        "dataset": "enron",
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
        "enron",
        instance,
        "| optimal t =", round(optimal_t, 6),
        "| max cut probability =", round(max_cut_probability, 6),
        "| support =", num_support
    )
df = pd.DataFrame(results)
df.to_csv("enron_fair_min_LP.csv",index=False)

resultsb = []
for file in files:

    P = np.load(file)
    instance = os.path.splitext(os.path.basename(file))[0]

    n = P.shape[0]
    edges = get_edge_list(P)
    weights = np.ones(len(edges))

     # Solve weighted balanced fair cut cover LP
    try:
        partitions, probs, cut_probs, optimal_t = (fair_max_bisection(edges, weights, n))
    except:
        print(f"Gurobi limitation exceeded on {instance}")
        continue

    # Weighted coverage of each hyperedge
    weighted_coverage = weights * cut_probs

    # Number of partitions assigned positive probability
    num_support = int(np.sum(probs > 1e-8))

    # Minimum unweighted cut probability
    min_cut_probability = float(np.min(cut_probs))

    resultsb.append({
        "dataset": "enron",
        "instance": instance,
        "n_vertices": n,
        "n_edges": len(edges),
        "optimal_t": optimal_t,
        "min_cut_probability": min_cut_probability,
        "num_support_partitions": num_support,
        "cut_probabilities": cut_probs,
        "weighted_coverages": weighted_coverage.tolist()
    })

    print(
        "enron",
        instance,
        "| optimal t =", round(optimal_t, 6),
        "| min cut probability =", round(min_cut_probability, 6),
        "| support =", num_support
    )
df = pd.DataFrame(resultsb)
df.to_csv("enron_fair_max_LP.csv",index=False)