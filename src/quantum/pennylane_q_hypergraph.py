import pickle
import networkx as nx
import pennylane as qml
from joblib import Parallel, delayed
from pennylane import numpy as np
from itertools import combinations

def make_M_V(P, w):
    M = P @ np.diag(w) @ P.T
    return M, np.diag(P @ w) - M

def max_cut_from_Q(Q, 
                   seed: int,
                   num_layers: int,
                   scaling_constant: float,
                   step_size: float, 
                   max_trials: int,
                   threshold: float,
                  ):
    rng = np.random.default_rng(seed=seed)
    n_wires = Q.shape[0]
    triu_indices = np.triu_indices(Q.shape[0])
    edge_list = []
    for (u, v) in zip(triu_indices[0], triu_indices[1]):
        if u == v:
            continue
        elif Q[u][v] != 0.0:
            edge_list.append((u, v))
    num_edges = len(edge_list)
    print("Number of edges: ", num_edges)
    beta_params = scaling_constant * rng.random([n_wires, num_layers], requires_grad=True)
    gamma_params = scaling_constant * rng.random([num_edges, num_layers], requires_grad=True)
    def U_B(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta[wire], wires=wire)
    def U_C(gamma):
        edge_idx = 0
        for u, v in edge_list: 
            w = Q[u][v]
            qml.CNOT(wires=(u, v))
            qml.RZ(-w * gamma[edge_idx], wires=v)
            qml.CNOT(wires=(u, v))
            edge_idx += 1 
    dev = qml.device('lightning.gpu', wires=n_wires)
    @qml.qnode(dev)
    def circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for k in range(num_layers):
            U_C(gammas[:, k])
            U_B(betas[:, k])
        return [qml.expval(qml.PauliZ(u) @ qml.PauliZ(v)) for u, v in edge_list]
    
    def objective(gammas, betas):
        expectation_values = circuit(gammas, betas)
        expectation_values = np.array(expectation_values)
        values = 0.0
        for u in range(n_wires):
            values += Q[u][u]
        edge_idx = 0
        for u, v in edge_list:
            w  = Q[u][v]
            values += 2 * expectation_values[edge_idx] * w
            edge_idx += 1
        return -values

    opt = qml.AdamOptimizer(stepsize=step_size)
    objective_list = [objective(gamma_params, beta_params)]
    params_list = [(gamma_params, beta_params)]
    for i in range(max_trials):
        (gamma_params, beta_params), current_obj = opt.step_and_cost(objective, gamma_params, beta_params)
        if (i + 1) % 50 == 0:
            print("Current obj: ", current_obj)
            if np.abs(current_obj - objective_list[-1]) <= threshold:
                objective_list.append(current_obj)
                params_list.append((gamma_params, beta_params))
                break
        objective_list.append(current_obj)
        params_list.append((gamma_params, beta_params))
    result = [objective_list, params_list]
    best_params = result[1][np.argmin(result[0])]
    best_qaoa_objective = np.min(result[0])
    
    print("Best param: ", best_params)
    print("Best QAOA obj: ", best_qaoa_objective)
    
    dev_sampling = qml.device('lightning.gpu', wires=n_wires)
    @qml.set_shots(shots = 10000)
    @qml.qnode(dev_sampling)
    def sampling_circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for k in range(num_layers):
            U_C(gammas[:, k])
            U_B(betas[:, k])
        return qml.sample()
    best_gammas, best_betas = best_params
    samples = sampling_circuit(best_gammas, best_betas)
    values = []
    formatted_samples = []
    for sample in samples:
        new_x = 2*np.array(sample)-1
        value = new_x@Q@new_x.transpose()
        formatted_samples.append(sample)
        values.append(value)
    print("Best objective:", np.max(values))
    print("Average objective:", np.mean(values))
    print("Best x:", formatted_samples[np.argmax(values)])
    return [np.mean(values), best_qaoa_objective, best_params] #np.max(values), formatted_samples[np.argmax(values)], 



def fair_cut_from_Q(Vlist, 
                   seed: int,
                   num_layers: int,
                   scaling_constant: float,
                   step_size: float, 
                   max_trials: int,
                   threshold: float,
                   #non_improving_steps: int
                   ):
    rng = np.random.default_rng(seed=seed)
    hyper_edges_list = []
    for hyper_edge_matrix in Vlist:
        hyper_edge = []
        hyper_edges_list.append(set(np.where(hyper_edge_matrix != 0.0)[1]))
    n_wires = Vlist[0].shape[0]
    edge_list = []
    hyper_edges_clique_enum = []
    hyper_edge_idx = 0
    for hyper_edge in hyper_edges_list:
        hyper_edge_expansion = []
        for edge in combinations(hyper_edge, 2):
            hyper_edge_expansion.append((np.min(edge), np.max(edge), float(Vlist[hyper_edge_idx][np.min(edge)][np.max(edge)])))
        edge_list += hyper_edge_expansion
        hyper_edges_clique_enum.append(hyper_edge_expansion)
        hyper_edge_idx += 1
    print("Len Edge list: ", len(edge_list))
    edge_list = set(edge_list)
    print("Len Set Edge list: ", len(edge_list))
    num_edges = len(edge_list)
    beta_params = scaling_constant * rng.random([n_wires, num_layers], requires_grad=True)
    gamma_params = scaling_constant * rng.random([num_edges, num_layers], requires_grad=True)
    def U_B(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta[wire], wires=wire)
    def U_C(gamma):
        edge_idx = 0
        for u, v, w in edge_list: 
            qml.CNOT(wires=(u, v))
            qml.RZ(-w * gamma[edge_idx], wires=v)
            qml.CNOT(wires=(u, v))
            edge_idx += 1
    dev = qml.device('lightning.gpu', wires=n_wires)
    @qml.qnode(dev)
    def circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for k in range(num_layers):
            U_C(gammas[:, k])
            U_B(betas[:, k])
        return [qml.expval(qml.PauliZ(u) @ qml.PauliZ(v)) for u, v, _ in edge_list]
    
    def objective(gammas, betas):
        tau = 0.005
        expectation_values = circuit(gammas, betas)
        exp_values_dict = {}
        edge_idx = 0
        for u, v, _ in edge_list:
            exp_values_dict[f"{u}-{v}"] = expectation_values[edge_idx]
            edge_idx += 1
        all_hyper_edges = []
        for hyper_edge_idx in range(len(hyper_edges_clique_enum)):
            hyper_edge_values = 0.0
            num_vertices = len(hyper_edges_list[hyper_edge_idx])
            for u, v, _ in hyper_edges_clique_enum[hyper_edge_idx]:
                hyper_edge_values += 2 * exp_values_dict[f"{u}-{v}"]
            value = (num_vertices - 1)/num_vertices  - (hyper_edge_values/(num_vertices)**2) #
            #value = np.exp(-value/tau)
            all_hyper_edges.append(value)
        return -np.min(all_hyper_edges) #tau*np.log(np.sum(all_hyper_edges)) #
        #values = np.min(all_hyper_edges)
        #return -values
    opt = qml.AdamOptimizer(stepsize=step_size)
    objective_list = [objective(gamma_params, beta_params)]
    params_list = [(gamma_params, beta_params)]
    for i in range(max_trials):
        (gamma_params, beta_params), current_obj = opt.step_and_cost(objective, gamma_params, beta_params)
        if (i + 1) % 50 == 0:
            print("Current obj: ", current_obj)
            if np.abs(current_obj - objective_list[-1]) <= threshold:
                objective_list.append(current_obj)
                params_list.append((gamma_params, beta_params))
                break
        objective_list.append(current_obj)
        params_list.append((gamma_params, beta_params))
    result = [objective_list, params_list]
    best_params = result[1][np.argmin(result[0])]
    best_qaoa_objective = np.min(result[0])
    
    num_shots = 10000
    dev_sampling = qml.device('lightning.gpu', wires=n_wires)
    @qml.set_shots(shots = num_shots)
    @qml.qnode(dev_sampling)
    def sampling_circuit(gammas, betas):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for k in range(num_layers):
            U_C(gammas[:, k])
            U_B(betas[:, k])
        return qml.counts()
    best_gammas, best_betas = best_params
    counts = sampling_circuit(best_gammas, best_betas)
    Qlist_values = []
    for Q in Vlist:
        Q_value = 0.0
        for sample in counts.keys():
            new_x = np.array([2*int(sample[i]) - 1 for i in range(len(sample))])
            value = new_x@Q@new_x.transpose()
            Q_value += (counts[sample]/num_shots)*value
        Qlist_values.append(Q_value)
    print("Best fair objective:", np.min(Qlist_values))
    return [np.min(Qlist_values), best_qaoa_objective, best_params]


def run_graph(idx, num_layers, run_type, graph_type):
    # if graph_type == "contact-high-school":
    #     if idx < 10: 
    #         hypergraph_name = f"contact-high-school_0{idx}"
    #     else:
    #         hypergraph_name = f"contact-high-school_{idx}"
    # elif graph_type == "email-Enron":
    #     hypergraph_name = f"email-Enron_{idx}"
    # elif graph_type == "congress-bills":
    #     if idx < 10:
    #         hypergraph_name = f"congress-bills_0{idx}"
    #     else:
    #         hypergraph_name = f"congress-bills_{idx}"
    if idx < 10:
        hypergraph_name = f"0{idx}"
    else:
        hypergraph_name = f"{idx}"
    print("Name: ", hypergraph_name)
    P = np.load(f'quantum_hypergraph/QuantumHypergraphPartitioning/data/{graph_type}/P/{hypergraph_name}.npy')
    w = np.load(f'quantum_hypergraph/QuantumHypergraphPartitioning/data/{graph_type}/w/{hypergraph_name}.npy')
    M, V = make_M_V(P, w)
    Vlist = []
    Q = V
    Vlist = [w[i] * (np.diag(P[:, i]) - P[:, i].reshape(P.shape[0], 1) * P[:, i].reshape(1, P.shape[0])) for i in range(P.shape[1])]
    if run_type == "max":
        step_size = 5e-2
    elif run_type == "fair":
        step_size = 5e-2
    threshold = 1e-8
    max_trials = 300
    scaling_constant = 5e-2
    seed = 42
    if run_type == "max":
        results  =  max_cut_from_Q(Q=Q,
                                    seed = seed,
                                    num_layers = num_layers,
                                    scaling_constant = scaling_constant,
                                    step_size = step_size, 
                                    max_trials = max_trials,
                                    threshold = threshold)
        print("Best QAOA obj: ", results[1])
        print("Best average max:", results[0])
    elif run_type == "fair":
        results  =  fair_cut_from_Q(Vlist = Vlist, #
                                    seed = seed,
                                    num_layers = num_layers,
                                    scaling_constant = scaling_constant,
                                    step_size = step_size, 
                                    max_trials = max_trials,
                                    threshold = threshold)
        print("Best QAOA obj: ", results[1])
        print("Best fair sampling:", results[0])
    with open(f'quantum_hypergraph/QuantumHypergraphPartitioning/quantum_results/correct_result/poisson/{run_type}_cut_{graph_type}_{hypergraph_name}_np15_ma_p={num_layers}_maxiter={max_trials}_seed={seed}.pkl', 'wb') as f:
        pickle.dump(results, f)