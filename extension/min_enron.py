from pennylane import numpy as np
import pennylane as qml
import itertools
from itertools import combinations
from scipy.optimize import minimize
from math import comb
import qaoa_aux 
#importlib.reload(qaoa_aux)
import os
import glob
import pandas as pd
folder_P = "/Users/stevehalley/Code/Github/QuantumHypergraphPartitioning/data/email-Enron/numpy_15"
def U_B(beta,edge_list):
    for edge_idx, (u, v) in enumerate(edge_list):
        theta = beta[edge_idx]
        qml.S(wires=u)
        qml.SingleExcitationMinus(-2 * theta,wires=[u, v])
        qml.adjoint(qml.S)(wires=u)
def U_B_standard(beta, edge_list):
    for u, v in edge_list:
        qml.S(wires=u)
        qml.SingleExcitationMinus(-2 * beta, wires=[u, v])
        qml.adjoint(qml.S)(wires=u)
def U_C_MC(gamma,edge_list):
    edge_idx = 0
    for u, v in edge_list: 
        
        qml.CNOT(wires=(u, v))
        qml.RZ(gamma[edge_idx], wires=v)
        qml.CNOT(wires=(u, v))
        edge_idx += 1  

def U_C(gamma,cost_terms):
    # we have term index + cost terms so all the even numbered ZZ's
    for term_idx, (_, wires, coeff) in enumerate(cost_terms):
        theta = 2 * gamma[term_idx] * coeff
        # example for 4 consectuve z's
        #qml.PauliRot(theta, "ZZZZ", wires=[0,1,2,3])
        qml.PauliRot(theta,"Z" * len(wires),wires=list(wires))

def objective(gammas,betas,circuit,hyperedge_list,cost_terms,weights):
    expectation_values = circuit(gammas, betas)
    value = 0.0

    # weighted constant term for each hyperedge
    for edge_idx, edge in enumerate(hyperedge_list):
        k = len(edge)
        #w_e * 1 - (1/2^k-1)
        value += weights[edge_idx] * (1 - 1 / (2 ** (k - 1)))

    # coeff is weighted in build_cost_terms()
    for term_idx, (_, _, coeff) in enumerate(cost_terms):
        value += coeff * expectation_values[term_idx]

    return value

def run_qaoa(
    hyperedge_list,
    n,
    weights,
    runtype,
    num_layers=1,
    seed=42,
    scaling_constant=0.1,
    step_size=0.01,
    max_trials=100
):
    n_wires = n
    assert len(weights) == len(hyperedge_list)

    # Build mixer edges and cost terms
    cost_edge_list = qaoa_aux.clique_expansion(hyperedge_list)
    mixer_edge_list = qaoa_aux.connected_clique_expansion(hyperedge_list, n)
    cost_terms = qaoa_aux.build_cost_terms(hyperedge_list,weights)

    # Balanced initial state
    target_weight = (n_wires + 1) // 2
    initial_state = qaoa_aux.make_dicke_state(n_wires, target_weight)

    dev = qml.device("lightning.qubit", wires=n_wires)
    
    @qml.qnode(dev, diff_method="adjoint",device_vjp=True)
    def aon_circuit(gammas, betas):
        qml.StatePrep(initial_state,wires=range(n_wires))

        for layer in range(num_layers):
            U_C(gammas[:, layer],cost_terms)
            U_B(betas[:, layer],mixer_edge_list)

        observables = []

        for _, wires, _ in cost_terms:
            z_term = qml.prod(*[qml.PauliZ(wire)for wire in wires])
            observables.append(qml.expval(z_term))
        return observables

    @qml.qnode(dev, diff_method="adjoint",device_vjp=True)
    def mc_circuit(gammas,betas):
        qml.StatePrep(initial_state,wires=range(n_wires))
        for layer in range(num_layers):
            U_C_MC(gammas[:,layer],cost_edge_list)
            U_B(betas[:, layer],mixer_edge_list)
        observables = []

        for _, wires, _ in cost_terms:
            z_term = qml.prod(*[qml.PauliZ(wire)for wire in wires])
            observables.append(qml.expval(z_term))
        return observables

    @qml.qnode(dev, diff_method="adjoint",device_vjp=True)
    def standard_circuit_aon(gammas, betas):
        qml.StatePrep(initial_state, wires=range(n_wires))    
        for layer in range(num_layers):
            # Multi-angle AoN cost
            U_C(gammas[:, layer], cost_terms)
            # Single-angle mixer
            U_B_standard(betas[layer], mixer_edge_list)
    
        observables = []
    
        for _, wires, _ in cost_terms:
            z_term = qml.prod(*[qml.PauliZ(wire) for wire in wires])
            observables.append(qml.expval(z_term))
    
        return observables

    @qml.qnode(dev, diff_method="adjoint",device_vjp=True)
    def standard_circuit_mc(gammas, betas):
        qml.StatePrep(initial_state, wires=range(n_wires))    
        for layer in range(num_layers):
            # single-angle AoN cost
            U_C_MC(gammas[:,layer],cost_edge_list)
            # Single-angle mixer
            U_B_standard(betas[layer], mixer_edge_list)
    
        observables = []
    
        for _, wires, _ in cost_terms:
            z_term = qml.prod(*[qml.PauliZ(wire) for wire in wires])
            observables.append(qml.expval(z_term))
    
        return observables

    

    if runtype == "aon":
        circuit = aon_circuit
        n_gamma = len(cost_terms)
        single_beta = False
    
    elif runtype == "mc":
        circuit = mc_circuit
        n_gamma = len(cost_edge_list)
        single_beta = False
    
    elif runtype == "standard_aon":
        circuit = standard_circuit_aon
        n_gamma = len(cost_terms)
        single_beta = True
    elif runtype == "standard_mc":
        circuit = standard_circuit_mc
        n_gamma = len(cost_edge_list)
        single_beta = True
    else:
        raise ValueError("runtype must be 'aon', 'mc','standard_aon', 'standard_mc'")

    # Random parameter initialization
    rng = np.random.default_rng(seed)
    
    gamma_params = np.array(scaling_constant * rng.random((n_gamma, num_layers)),requires_grad=True)
    # Single-angle or multi-angle mixer parameters
    if single_beta:
        beta_shape = (num_layers,)
    else:
        beta_shape = (len(mixer_edge_list), num_layers)
    
    beta_params = np.array(scaling_constant * rng.random(beta_shape),requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=step_size)
    objective_values = []
    print("Gamma shape:", gamma_params.shape)
    print("Beta shape:", beta_params.shape)
    print("Total parameters:", gamma_params.size + beta_params.size)
    
    # Define optimization objective
    def loss(g, b):
        return objective(
            g,
            b,
            circuit,
            hyperedge_list,
            cost_terms,
            weights
        )
    
    initial_value = loss(gamma_params, beta_params)
    objective_values.append(float(initial_value))
    
    print(f"Initial objective = {initial_value:.6f} | Runtype = {runtype}")
    
    # Optimization
    for trial in range(max_trials):
    
        gamma_params, beta_params = opt.step(
            loss,
            gamma_params,
            beta_params
        )
    
        current_value = loss(gamma_params, beta_params)
        objective_values.append(float(current_value))
    
        if trial % 50 == 0:
            print(
                f"Iteration {trial:3d} | "
                f"objective = {current_value:.6f}"
            )
    
    return {
        "gammas": gamma_params,
        "betas": beta_params,
        "objective_values": objective_values,
        "circuit": circuit,
        "cost_edge_list": cost_edge_list,
        "mixer_edge_list": mixer_edge_list,
        "cost_terms": cost_terms,
        "n_wires": n_wires
    }

def get_edge_list(P):
    edges=[]
    for idx in range(P.shape[1]):
        #indices that are not 0
        edge = tuple(np.flatnonzero(P[:,idx]))
        edges.append(edge)
    return edges

p_files = sorted(glob.glob(f"{folder_P}/*.npy"))


results_aon = []
results_mc=[]

for p_file in p_files:

    instance = os.path.splitext(os.path.basename(p_file))[0]
    # Load hypergraph
    P = np.load(p_file)
    edges = get_edge_list(P)
    w = np.ones(len(edges))
    hyperedge_list = qaoa_aux.matrix_to_hyperedges(P)
    n = P.shape[0]

    #circuit depth
    for p in [3]:

        print(f"\nRunning instance {instance} | p={p} | Runtype=AON")

        result_aon = run_qaoa(hyperedge_list,weights=w,runtype="aon",n=n,num_layers=p,seed=42,scaling_constant=5e-2,step_size=0.01,max_trials=300)
        final_objective = float(result_aon["objective_values"][-1])

        initial_objective = float(result_aon["objective_values"][0])

        results_aon.append({
            "dataset": "enron",
            "instance": instance,
            "runtype": "AoN",
            "p": p,
            "n_vertices": n,
            "n_hyperedges": len(hyperedge_list),
            "gammas": np.array2string(np.asarray(result_aon["gammas"]),separator=","),
            "betas": np.array2string(np.asarray(result_aon["betas"]),separator=","),
            "initial_objective": initial_objective,
            "final_objective": final_objective 
        })
        
df_aon = pd.DataFrame(results_aon)
# df_mc = pd.DataFrame(results_mc)
csv_path_aon = "enron_min_bisection_aon.csv"
# csv_path_mc = "poisson_16_12_4_min_bisection_mc.csv"

df_aon.to_csv(csv_path_aon,index=False)
# df_mc.to_csv(csv_path_mc,index=False)

print(f"saved results to: {csv_path_aon}")
