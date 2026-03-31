using PyCall

numpy = pyimport("numpy")

include("../src/classical/exact.jl")
include("../src/classical/sdp.jl")

function run_tests()
    for dataset in ["contact-high-school", "email-Enron"]
        for f in readdir("data/$dataset/numpy/")
            P = Matrix(numpy.load("data/$dataset/numpy/$f"))
            println(P)
        end
    end
end
