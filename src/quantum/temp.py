from math import comb
from pennylane import numpy as np
from itertools import combinations

def make_dicke_state(n, k):
        #generally k will be half of n for balanced partition
        if k < 0 or k > n:
            raise ValueError("k must satisfy 0 <= k <= n")
    
        state = np.zeros(2**n, dtype=complex)
    
        # All choices of k wires that contain a 1
        weight_k_states = list(combinations(range(n), k))
        # equal emplitude for each state
        amplitude = 1 / np.sqrt(len(weight_k_states))

        # will have sets of bits now ex: for n=4 and k=2 --> (0,1)(0,2),(0,3),(1,2),(1,3),(2,3) --> set a one at each of these index
        #(0,1) -> |1100> -> index 12
        #index 11 skipped so that amplitude stays at 0
        #(0,2) -> |1010> -> index 10
        
        for one_positions in weight_k_states:
            basis_index=0
            #wire index to statevector index
            for wire in one_positions:
                basis_index+=2**(n-1-wire)
            #basis_index = sum(2 ** (n - 1 - wire) for wire in one_positions)
            state[basis_index] = amplitude

        
        print(state)
        d_state = np.array(state, requires_grad=False)
        if (comb(n,k)!=np.count_nonzero(d_state)):
            raise ValueError("amplitude count incorrect")
    
        return d_state

state = make_dicke_state(4,2)
print(len(state))
