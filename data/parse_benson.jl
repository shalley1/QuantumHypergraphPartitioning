using SparseArrays, KaHyPar, PyCall, LinearAlgebra, Laplacians

hgraph_classes = ["congress-bills"] # ["NDC-classes", "email-Enron", "contact-high-school"]

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
    simps = Set()
    rows = rowvals(inc)
    for c in axes(inc, 2)
        e = rows[nzrange(inc, c)]
        issubset(e, part) || continue
        if length(e) > 1
            push!(simps, e)
        end
    end
    vmap = Dict()
    for (i, v) in enumerate(part)
        vmap[v] = i
    end
    P = stochastic_incidence([[vmap[x] for x in s] for s in simps])
    return stochastic_incidence([[vmap[x] for x in s] for s in simps])
end

function get_subgraphs(inc, target, imb)
    n, _ = size(inc)
    h = hypergraph(inc)
    nump = floor(Int, n / target)
    parts = KaHyPar.partition(h, nump; configuration=:connectivity, imbalance=imb)
    return filter(x -> nnz(x) != 0, [get_subgraph(inc, findall(x -> x == i, parts)) for i in 0:nump])
end

function gen_graph(target)
    for f in hgraph_classes
        simps = parser(f)
        inc = stochastic_incidence(simps)
        sgs = get_subgraphs(inc, target, 0.3)
        sgs = [A[reshape(sum(A; dims=2) .> 0, :), :] for A in sgs]
        sgs = collect(filter(x -> size(x, 1) <= target, sgs))
        mkpath("data/$f/numpy_$target/")
        pad = length(string(length(sgs)))
        for (i, sg) in enumerate(sgs)
            fname = "$(f)_$(lpad(string(i), pad, "0"))"
            println(fname, " ", size(sg), " ", nnz(sg), " ", (nnz(sg * sg') - size(sg, 1)) ÷ 2)
            numpy.save("data/$f/numpy_$target/$fname.npy", Matrix(sg), false)
        end
    end
end

function gen_sparse(target)
    for dataset in hgraph_classes
        for f in readdir("data/$dataset/numpy/")
            P = Matrix(numpy.load("data/$dataset/numpy/$f"))
            w = ones(size(P, 2))
            M = imb(P, w)
            M[diagind(M)] .= 0
            M2 = sparsify(sparse(M))
            numpy.save("data/$dataset/numpy/sparse_$f", Matrix(M2), false)
        end
    end
end

function edge_counts()
    for dataset in hgraph_classes
        for f in readdir("data/$dataset/numpy/")
            P = sparse(Matrix(numpy.load("data/$dataset/numpy/$f")))
            w = ones(size(P, 2))
            # println("$f $(size(P)) $((nnz(P * P') - size(P, 1)) ÷ 2)")
            a = [collect(nzrange(sparse(P), i)) for i in axes(P, 2)]
            acc = 0
            for i in axes(a, 1)
                for j in axes(a, 2)
                    if i != j && issubset(a[i], a[j])
                        acc += 1
                    end
                end
            end
            println("$f $acc")
            # for i in [length(nzrange(sparse(P), i)) for i in axes(P, 2)]
            #     println("$f $i")
            # end
        end
    end
end

function gen_pareto()
    P = Matrix(numpy.load("data/email-Enron/numpy_15/email-Enron_5.npy"))
    w = ones(size(P, 2))
    P1, w1 = P, w
    P2, w2 = ones(size(P1, 1), 1) ./ size(P1, 1), [1]
    M = imb(P1, w1)
    V = var(P2, w2)
    mkpath("data/pareto/email-Enron_05/combined")
    numpy.save("data/pareto/email-Enron_05/V.npy", Matrix(V), false)
    numpy.save("data/pareto/email-Enron_05/M.npy", Matrix(M), false)
    nsteps = 20

    for (i, α) in enumerate(range(0, 1; length=nsteps))
        O = α .* M ./ eigmax(M) + (1 - α) .* V ./ eigmax(V)
        f = "$(lpad(string(i), length(string(nsteps)), "0")).npy"
        numpy.save("data/pareto/email-Enron_05/combined/$f", O, false)
    end
end