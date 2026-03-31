using SparseArrays, KaHyPar, PyCall, LinearAlgebra

numpy = pyimport("numpy")

function parser(folder)
    return open("data/$(folder)/$(folder)-simplices.txt") do f
        simps = Set{Vector{Int}}()
        idx = 0
        for n in map(x -> parse(Int, x), readlines("data/$(folder)/$(folder)-nverts.txt"))
            simp = Int[]
            for _ in 1:n
                idx += 1
                push!(simp, parse(Int, readline(f)))
            end
            push!(simps, simp)
        end
        return simps
    end
end

function stochastic_incidence(simps)
    V, E, W = Int[], Int[], Float64[]
    for (e, s) in enumerate(simps)
        for v in s
            push!(V, v), push!(E, e), push!(W, 1 / length(s))
        end
    end
    return sparse(V, E, W)
end

function get_subgraph(inc, part)
    n = size(inc, 1)
    mask1 = zeros(n)
    mask1[part] .= 1
    acc = ceil.(Int, inc)' * mask1
    es = findall(x -> x > 1, acc)
    return (inc/Diagonal(acc))[part, es]
end

function get_subgraphs(inc, target, imb)
    n, _ = size(inc)
    h = hypergraph(inc)
    nump = floor(Int, n / target)
    parts = KaHyPar.partition(h, nump; configuration=:connectivity, imbalance=imb)
    return filter(x -> nnz(x) != 0, [get_subgraph(inc, findall(x -> x == i, parts)) for i in 0:nump])
end


function gen_graph()
    for f in ["contact-high-school", "email-Enron"]
        simps = parser(f)
        inc = stochastic_incidence(simps)
        sgs = get_subgraphs(inc, 15, 0.1)
        for (i, sg) in enumerate(sgs)
            numpy.save("data/$f/numpy/$(f)_$i.npy", Matrix(sg), false)
        end
    end
end
