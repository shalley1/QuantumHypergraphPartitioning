import argparse
from pennylane import numpy as np
import pennylane as qml
import itertools
from itertools import combinations
from scipy.optimize import minimize
from math import comb
import sys
sys.path.append("../")
import qaoa_aux 
import os
import glob
import pandas as pd

import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="Run QAOA experiments")

    # Dataset
    ap.add_argument("--dataset", type=str, choices=["poisson", "enron", "congress"], required=True)
    ap.add_argument("--output",type=str,required=True)

    # QAOA circuit configuration
    ap.add_argument("--runtype", type=str, choices=["aon", "mc", "standard_aon", "standard_mc"], default="aon")
    ap.add_argument("--loss_type", type=str, choices=["max", "min", "fair_max", "fair_min"], default="max")
    ap.add_argument("--mixer_graph", type=str, choices=["complete", "connected"], default="complete")

    # Optimization parameters
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaling_constant", type=float, default=5e-2)
    ap.add_argument("--step_size", type=float, default=0.01)
    ap.add_argument("--max_trials", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    # Device
    ap.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")

    return ap.parse_args()



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

def U_C_AoN(gamma,cost_terms):
    # we have term index + cost terms so all the even numbered ZZ's
    for term_idx, (_, wires, coeff) in enumerate(cost_terms):
        theta = 2 * gamma[term_idx] * coeff
        # example for 4 consectuve z's
        #qml.PauliRot(theta, "ZZZZ", wires=[0,1,2,3])
        # pennylane will decompose this into CNOT+RZ ladder 
        qml.PauliRot(theta,"Z" * len(wires),wires=list(wires))


def max_bi_objective(gammas,betas,circuit,hyperedge_list,cost_terms,weights):
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

def min_bi_objective(gammas,betas,circuit,hyperedge_list,cost_terms,weights):
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

def fair_max_objective(gammas, betas, circuit,hyperedge_list, cost_terms, weights):
    expectation_values = circuit(gammas, betas)

    # Store weighted expected cut value for each hyperedge
    edge_values = []
    for edge_idx, edge in enumerate(hyperedge_list):
        k = len(edge)
        # Weighted constant term for this hyperedge
        edge_value = weights[edge_idx] * (1 - 1 / (2 ** (k - 1)))

        # Add the Pauli-Z terms belonging to this hyperedge
        for term_idx, (source_edge_idx, wires, coeff) in enumerate(cost_terms):

            if source_edge_idx == edge_idx:
                edge_value += coeff * expectation_values[term_idx]

        edge_values.append(edge_value)

    # Negative because Adam minimizes
    return -qml.math.min(qml.math.stack(edge_values))

def fair_min_objective(gammas, betas, circuit,hyperedge_list, cost_terms, weights):
    expectation_values = circuit(gammas, betas)

    # Store weighted expected cut value for each hyperedge
    edge_values = []
    for edge_idx, edge in enumerate(hyperedge_list):
        k = len(edge)
        # Weighted constant term for this hyperedge
        edge_value = weights[edge_idx] * (1 - 1 / (2 ** (k - 1)))

        # Add the Pauli-Z terms belonging to this hyperedge
        for term_idx, (source_edge_idx, wires, coeff) in enumerate(cost_terms):
            if source_edge_idx == edge_idx:
                edge_value += coeff * expectation_values[term_idx]
        edge_values.append(edge_value)

    return qml.math.max(qml.math.stack(edge_values))


def run_qaoa(
    hyperedge_list,
    n,
    weights,
    runtype,
    loss_type="max",
    mixer_graph="complete",
    num_layers=3,
    seed=42,
    scaling_constant=5e-2,
    step_size=0.01,
    max_trials=300,
    device = "cpu",
    patience=50,
    min_delta=1e-3,
):


    n_wires = n
    assert len(weights) == len(hyperedge_list)


    cost_edge_list = qaoa_aux.clique_expansion(hyperedge_list)
    cost_terms = qaoa_aux.build_cost_terms(hyperedge_list,weights)
    if mixer_graph=="complete": 
        mixer_edge_list = qaoa_aux.complete_graph_mixer(n)
    else:
        mixer_edge_list = qaoa_aux.connected_clique_expansion(hyperedge_list,n)



    # Balanced initial state
    target_weight = (n_wires + 1) // 2
    initial_state = qaoa_aux.make_dicke_state(n_wires, target_weight)

    if device=="cpu":
        dev = qml.device("lightning.qubit", wires=n_wires)

    else:
        dev = qml.device("lightning.gpu",wires=n_wires)


    @qml.qnode(dev, diff_method="adjoint",device_vjp=True)
    def aon_circuit(gammas, betas):
        qml.StatePrep(initial_state,wires=range(n_wires))

        for layer in range(num_layers):
            U_C_AoN(gammas[:, layer],cost_terms)
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
            U_C_AoN(gammas[:, layer], cost_terms)
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

    # Select objective function
    if loss_type == "max":
        objective = max_bi_objective
        sign = -1

    elif loss_type == "min":
        objective = min_bi_objective
        sign = 1

    elif loss_type == "fair_max":
        objective = fair_max_objective
        sign = 1

    elif loss_type == "fair_min":
        objective = fair_min_objective
        sign = 1

    else:
        raise ValueError("Invalid loss_type")


    def loss(g, b):
        return sign * objective(
            g,
            b,
            circuit,
            hyperedge_list,
            cost_terms,
            weights
        )
    initial_value = loss(gamma_params, beta_params)

    def actual_objective(loss_value):
        if loss_type in ("max", "fair_max"):
            return -loss_value
        return loss_value

    objective_values.append(float(actual_objective(initial_value)))

    best_loss = float(initial_value)
    best_gammas = gamma_params.copy()
    best_betas = beta_params.copy()

    wait = 0
    stopped_early = False

    print(f"Initial objective = {actual_objective(initial_value):.6f}")
    print(f"Runtype = {runtype} | Loss = {loss_type}")
    best_loss_for_patience = best_loss
    for trial in range(max_trials):
        gamma_params, beta_params = opt.step(
            loss, gamma_params, beta_params
        )

        current_value = loss(gamma_params, beta_params)
        current_loss = float(current_value)
        actual_value = actual_objective(current_value)

        objective_values.append(float(actual_value))

        # Always retain the best solution encountered
        if current_loss < best_loss:
            best_loss = current_loss
            best_gammas = gamma_params.copy()
            best_betas = beta_params.copy()

        # Reset patience only when improvement is significant
        if current_loss < best_loss_for_patience - min_delta:
            best_loss_for_patience = current_loss
            wait = 0
        else:
            wait += 1

        if trial % 10 == 0:
            print(
                f"Iteration {trial:3d} | "
                f"objective = {actual_value:.6f} | "
                f"patience = {wait}/{patience}"
            )

        if wait >= patience:
            stopped_early = True
            print(f"Early stopping at iteration {trial + 1}")
            break

    gamma_params = best_gammas
    beta_params = best_betas   
    return {
        "gammas": gamma_params,
        "betas": beta_params,
        "num_layers":num_layers,
        "objective_values": objective_values,
        "circuit": circuit,
        "runtype":runtype,
        "objective":loss_type,
        "mixer_graph":mixer_graph,
        "cost_edge_list": cost_edge_list,
        "mixer_edge_list": mixer_edge_list,
        "cost_terms": cost_terms,
        "n_wires": n_wires,
        "iterations_run": len(objective_values) - 1,
        "stopped_early": stopped_early,
        "best_objective": float(actual_objective(best_loss)),
    }
from pathlib import Path

def load_instances(dataset):
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if dataset == "poisson":
        datasets = [
            ("poisson_16_12_3", 20),
            ("poisson_16_12_4", 10),
            ("poisson_16_12_5", 10)
        ]

        for folder_name, limit in datasets:
            folder_P = data_dir / folder_name / "P"
            folder_w = data_dir / folder_name / "w"

            p_files = sorted(folder_P.glob("*.npy"))[:limit]

            for p_file in p_files:
                instance = p_file.stem
                P = np.load(p_file)
                w = np.load(folder_w / f"{instance}.npy")

                hyperedge_list = qaoa_aux.matrix_to_hyperedges(P)
                n = P.shape[0]

                yield folder_name, instance, hyperedge_list, n, w

    elif dataset == "enron":
        folder_P = data_dir / "email-Enron" / "numpy_15"
        p_files = sorted(folder_P.glob("*.npy"))

        for p_file in p_files:
            instance = p_file.stem
            P = np.load(p_file)

            n = P.shape[0]

            hyperedge_list = [
                [int(v) for v in np.flatnonzero(P[:, j])]
                for j in range(P.shape[1])
            ]

            if any(len(edge) == 0 for edge in hyperedge_list):
                raise ValueError(f"{instance} contains empty hyperedges")

            w = np.ones(len(hyperedge_list))

            yield "enron", instance, hyperedge_list, n, w

    elif dataset == "congress":
        folder_P = data_dir / "congress-bills" / "numpy_20"
        p_files = sorted(folder_P.glob("*.npy"))[:10]

        for p_file in p_files:
            instance = p_file.stem
            P = np.load(p_file)

            hyperedge_list = qaoa_aux.matrix_to_hyperedges(P)
            n = P.shape[0]
            w = np.ones(len(hyperedge_list))

            yield "congress", instance, hyperedge_list, n, w

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    args = parse_args()

    print("Dataset:", args.dataset)
    print("Runtype:", args.runtype)
    print("Loss:", args.loss_type)
    print("Mixer:", args.mixer_graph)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for dataset_name, instance, hyperedge_list, n, weights in load_instances(args.dataset):

        print("=" * 60)
        print(f"Running {instance} | Dataset={dataset_name}")
        print(f"Vertices={n} | Hyperedges={len(hyperedge_list)}")
        print(f"Runtype={args.runtype} | Loss={args.loss_type}")
        print("=" * 60)

        result = run_qaoa(
            hyperedge_list=hyperedge_list,
            n=n,
            weights=weights,
            patience=args.patience,
            min_delta=args.min_delta,
            runtype=args.runtype,
            loss_type=args.loss_type,
            mixer_graph=args.mixer_graph,
            num_layers=args.num_layers,
            seed=args.seed,
            scaling_constant=args.scaling_constant,
            step_size=args.step_size,
            max_trials=args.max_trials,
            device=args.device
        )

        row = {
            "dataset": dataset_name,
            "instance": instance,
            "n": n,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "iterations_run": result["iterations_run"],
            "stopped_early": result["stopped_early"],
            "best_objective": result["best_objective"],
            "num_hyperedges": len(hyperedge_list),
            "runtype": args.runtype,
            "loss_type": args.loss_type,
            "mixer_graph": args.mixer_graph,
            "num_layers": args.num_layers,
            "seed": args.seed,
            "scaling_constant": args.scaling_constant,
            "step_size": args.step_size,
            "max_trials": args.max_trials,
            "initial_objective": result["objective_values"][0],
            "final_objective": result["best_objective"],
        }

        df = pd.DataFrame([row])

        # Append results after each instance
        df.to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False
        )

        print(f"Saved {instance} to {output_file}")

    print("All experiments completed!")
