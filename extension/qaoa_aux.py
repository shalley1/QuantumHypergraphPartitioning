from pennylane import numpy as np
import pennylane as qml
import itertools
from itertools import combinations
from scipy.optimize import minimize
from math import comb

def make_dicke_state(n, k):
        if k < 0 or k > n:
            raise ValueError("k must satisfy 0 <= k <= n")
    
        state = np.zeros(2**n, dtype=complex)
        weight_k_states = list(combinations(range(n), k))
        
        amplitude = 1 / np.sqrt(len(weight_k_states))

        #convert bit string to integer
        for one_positions in weight_k_states:
            basis_index=0
            for wire in one_positions:
                basis_index+=2**(n-1-wire)
            state[basis_index] = amplitude

        

        d_state = np.array(state, requires_grad=False)
        if (comb(n,k)!=np.count_nonzero(d_state)):
            raise ValueError("amplitude count incorrect")
    
        return d_state

def clique_expansion(hyperedges):
    mixer_edges = set()

    for edge in hyperedges:
        for u, v in itertools.combinations(edge, 2):
            mixer_edges.add(tuple(sorted((u, v))))

    return sorted(mixer_edges)
def build_cost_terms(hyperedge_list,weights):
    """
    Returns a list of expanded even-order Z terms.

    Each entry:
        (hyperedge_idx, wires, coefficient) w
    """
    cost_terms = []
    assert len(weights) == len(hyperedge_list)

    for edge_idx, edge in enumerate(hyperedge_list):
        edge = tuple(edge)
        k = len(edge)

        # All nonconstant terms have the same coefficient
        coeff = -weights[edge_idx] / (2 ** (k - 1))

        # Keep only even-order Z products: 2, 4, 6, ...
        for order in range(2, k + 1, 2):
            for wires in combinations(edge, order):
                cost_terms.append(
                    (edge_idx, wires, coeff)
                )

    return cost_terms

def matrix_to_hyperedges(P):
    hyperedge_list = []

    for edge_idx in range(P.shape[1]):
        vertices = np.nonzero(P[:, edge_idx])[0]
        hyperedge_list.append(vertices.tolist())

    return hyperedge_list