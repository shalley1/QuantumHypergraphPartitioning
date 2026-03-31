using JuMP
import Clarabel
using Random
using LinearAlgebra


function svd_cholesky(X::AbstractMatrix)
    F = LinearAlgebra.svd(X)
    D = LinearAlgebra.Diagonal(sqrt.(F.S))
    return D * F.U'
end

function max_sdp(Q)
    N = size(Q, 2)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, X[1:N, 1:N], PSD)
    for i in 1:N
        set_start_value(X[i, i], 1.0)
    end
    @objective(model, Max, LinearAlgebra.dot(Q, X))
    @constraint(model, LinearAlgebra.diag(X) .== 1)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return svd_cholesky(value(X))
end

function min_sdp(Q1, Q2, beta)
    N = size(Q1, 2)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, X[1:N, 1:N], PSD)
    for i in 1:N
        set_start_value(X[i, i], 1.0)
    end
    @objective(model, Min, LinearAlgebra.dot(Q1, X))
    @constraint(model, LinearAlgebra.diag(X) .== 1)
    @constraint(model, LinearAlgebra.dot(Q2, X) <= beta)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return svd_cholesky(value(X))
end

function min_max_expected(Qs; maxmin=false)
    N = size(Qs[1], 2)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, X[1:N, 1:N], PSD)
    @constraint(model, diag(X) .== 1)
    for i in 1:N
        set_start_value(X[i, i], 1.0)
    end

    if maxmin
        @variable(model, t >= 0)
        @objective(model, Max, t)
        for Q in Qs
            @constraint(model, dot(Q, X) >= t)
        end
    else
        @variable(model, t >= 0)
        @objective(model, Min, t)
        for Q in Qs
            @constraint(model, dot(Q, X) <= t)
        end
    end
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return svd_cholesky(value(X))
end

hr(M, nsamples) = sign.(M * randn(size(M, 2), nsamples))

function best_hr(Q, X, nsamples, maximize=false)
    S = hr(X, nsamples)
    f = maximize ? maximum : minimum
    return f(dot(Q * S[:, i], S[:, i]) for i in 1:nsamples)
end

function sdp_minmax(Qs, M; maxmin=false)
    f = maxmin ? maximum : minimum
    return f(dot(Q, asin.(M * M') / π) for Q in Qs)
end
