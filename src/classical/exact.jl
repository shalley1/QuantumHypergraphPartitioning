using Combinatorics, JuMP, Gurobi

function max_qp(Q)
    N = size(Q, 2)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, x[1:N], Binary)
    y = 2x .- 1
    @objective(model, Min, dot(Q * y, y))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return value(x)
end

function min_qp(Q1, Q2)
    N = size(Q1, 2)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, x[1:N], Binary)
    @objective(model, Min, dot(Q1 * y, y))
    @constraint(model, dot(Q2 * y, y) <= beta)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return value(x)
end

function minmax_expectation(Qs; maxmin=false)
    n = size(Qs[1], 2)
    m = length(Qs)
    Y = zeros(m, 2^n)
    for (j, s) in enumerate(powerset(1:n))
        x = ones(n)
        x[s] *= -1
        for (i, Q) in enumerate(Qs)
            Y[i, j] = dot(Q * x, x)
        end
    end
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, p[1:2^n] >= 0)
    @variable(model, t)
    if maxmin
        @objective(model, max, t)
        @constraint(model, Y * p .>= t)
        @constraint(model, sum(p) == 1)
    else
        @objective(model, Min, t)
        @constraint(model, Y * p .<= t)
        @constraint(model, sum(p) == 1)
    end
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return value(p), value(t)
end
