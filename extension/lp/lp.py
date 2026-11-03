import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import glob
import pandas as pd


"""
the below functions are LP solvers for the following set of problems on balanced hypergraphs:
    1. Max Bisection
    2. Min Bisection
    3. Fair Max Bisection (Fair cut cover)
    4. Fair Min Bisection
"""


def min_bisection(n, edges,weights):
    assert len(weights) == len(edges)
    model = gp.Model()
    model.Params.OutputFlag = 0

    # x_i in {0,1}
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # c_e = 1 if hyperedge e is cut
    c = model.addVars(len(edges), vtype=GRB.BINARY, name="cut")

    target_ones = (n + 1) // 2
    # exact balance: half the vertices in partition 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) ==target_ones,name="balance")

    # AoN cut constraints
    for e_idx, edge in enumerate(edges):
        for u, v in itertools.combinations(edge, 2):
            #enforce that the edge is cut if vertexes in different partitions
            #essentially implementation of max |x_u-x_v| ==L(x) so cost must be 1 or 0
            model.addConstr(c[e_idx] >= x[u] - x[v])
            model.addConstr(c[e_idx] >= x[v] - x[u])

    # minimize total number of cut hyperedges
    objective = gp.quicksum(weights[e_idx]*c[e_idx] for e_idx in range(len(edges)))

    model.setObjective(objective, GRB.MINIMIZE)

    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Gurobi did not prove optimality. Status = {model.Status}"
    # print(x)
    # print(c)
    x_opt = np.array([int(round(x[i].X))for i in range(n)])
    cut_opt = np.array([int(round(c[e_idx].X)) for e_idx in range(len(edges))])


    return x_opt, cut_opt, model.ObjVal
    


def max_bisection(n, edges,weights):
    assert len(weights) == len(edges)
    model = gp.Model()
    model.Params.OutputFlag = 0

    # x_i in {0,1}
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # c_e = 1 if hyperedge e is cut
    c = model.addVars(len(edges), vtype=GRB.BINARY, name="cut")

    target_ones = (n + 1) // 2
    # exact balance: half the vertices in partition 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) ==target_ones,name="balance")

    # AoN cut constraints
    for e_idx, edge in enumerate(edges):
        for u, v in itertools.combinations(edge, 2):
            #enforce that the edge is cut if vertexes in different partitions
            #essentially implementation of max |x_u-x_v| ==L(x) so cost must be 1 or 0
            model.addConstr(c[e_idx] >= x[u] - x[v])
            model.addConstr(c[e_idx] >= x[v] - x[u])

        #add in upper bounds        
        model.addConstr(c[e_idx] <= gp.quicksum(x[u] for u in edge))
        model.addConstr(c[e_idx] <= gp.quicksum(1 - x[u] for u in edge))
        

    # minimize total number of cut hyperedges
    objective = gp.quicksum(weights[e_idx]*c[e_idx] for e_idx in range(len(edges)))

    model.setObjective(objective, GRB.MAXIMIZE)

    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Gurobi did not prove optimality. Status = {model.Status}"
    # print(x)
    # print(c)
    x_opt = np.array([int(round(x[i].X))for i in range(n)])
    cut_opt = np.array([int(round(c[e_idx].X)) for e_idx in range(len(edges))])


    return x_opt, cut_opt, model.ObjVal

def fair_max_bisection(hyperedges,weights,n):
    assert len(weights)==len(hyperedges)
    model = gp.Model()
    model.Params.OutputFlag=0

    target_ones = (n+1)//2
    partitions=[]

    for ones in itertools.combinations(range(n),target_ones):
        x = np.zeros(n,dtype=int)
        for one in ones: 
            x[one]=1
        partitions.append(x)

    p = model.addVars(len(partitions),lb=0,ub=1,name="p")
    t = model.addVar(lb=0,name='t')

    model.addConstr(
        gp.quicksum(p[j] for j in range(len(partitions))) == 1,
        name="prob_normalization"
    )#sum probabilities=1
    
    # Weighted fair cut constraints
    for e_idx, edge in enumerate(hyperedges):
        expected_cut = 0
        for j, partition in enumerate(partitions):
            vertex_assignments = [partition[v] for v in edge]#list of the vertices in each hyperedge
            is_cut = int(len(set(vertex_assignments)) > 1)
            expected_cut += p[j] * is_cut 
        # weighted probability of cutting this hyperedge >= t
        model.addConstr(weights[e_idx] * expected_cut >= t,name=f"coverage_{e_idx}")
    model.setObjective(t,GRB.MAXIMIZE)
    model.optimize()
    assert model.Status == GRB.OPTIMAL, (f"Gurobi did not prove optimality. Status = {model.Status}")
    probs =np.array([p[j].X for j in range(len(partitions))]) #get solution value from gurobi 
    cut_probabilities = []

    # Calculate the probability of cutting each hyperedge
    for edge in hyperedges:
        cut_probability = 0
        for j, partition in enumerate(partitions):
            vertex_assignments = [partition[v] for v in edge]
            is_cut = int(len(set(vertex_assignments)) > 1)
            cut_probability += probs[j] * is_cut
    
        cut_probabilities.append(cut_probability)

    return (np.array(partitions),probs,cut_probabilities,model.ObjVal) 
    
def fair_min_bisection(hyperedges,weights,n):
    assert len(weights)==len(hyperedges)
    model = gp.Model()
    model.Params.OutputFlag=0

    target_ones = (n+1)//2
    partitions=[]

    for ones in itertools.combinations(range(n),target_ones):
        x = np.zeros(n,dtype=int)
        for one in ones: 
            x[one]=1
        partitions.append(x)

    p = model.addVars(len(partitions),lb=0,ub=1,name="p")
    t = model.addVar(lb=0,name='t')

    model.addConstr(
        gp.quicksum(p[j] for j in range(len(partitions))) == 1,
        name="prob_normalization"
    )#sum probabilities=1
    
    # Weighted fair cut constraints
    for e_idx, edge in enumerate(hyperedges):
        expected_cut = 0
        for j, partition in enumerate(partitions):
            vertex_assignments = [partition[v] for v in edge]#list of the vertices in each hyperedge
            is_cut = int(len(set(vertex_assignments)) > 1)
            expected_cut += p[j] * is_cut 
        # weighted probability of cutting this hyperedge <= t
        model.addConstr(weights[e_idx] * expected_cut <= t,name=f"coverage_{e_idx}")
    model.setObjective(t,GRB.MINIMIZE)
    model.optimize()
    assert model.Status == GRB.OPTIMAL, (f"Gurobi did not prove optimality. Status = {model.Status}")
    probs =np.array([p[j].X for j in range(len(partitions))]) #get solution value from gurobi 
    cut_probabilities = []

    # Calculate the probability of cutting each hyperedge
    for edge in hyperedges:
        cut_probability = 0
        for j, partition in enumerate(partitions):
            vertex_assignments = [partition[v] for v in edge]
            is_cut = int(len(set(vertex_assignments)) > 1)
            cut_probability += probs[j] * is_cut
    
        cut_probabilities.append(cut_probability)

    return (np.array(partitions),probs,cut_probabilities,model.ObjVal) 
    
    