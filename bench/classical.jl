using PyCall

numpy = pyimport("numpy")

include("../src/classical/exact.jl")
include("../src/classical/sdp.jl")

function run_tests()
    for dataset in ["contact-high-school", "email-Enron"]
        for f in readdir("data/$dataset/numpy/")
            P = Matrix(numpy.load("data/$dataset/numpy/$f"))
            M = P * P'
            println(size(M))
            println(size(P))
            V = Diagonal(sum(P; dims=2)) - M
            X = max_sdp(V)
            res = best_hr(V, X, 1000, maximize=true)
            println(res)
        end
    end
end
