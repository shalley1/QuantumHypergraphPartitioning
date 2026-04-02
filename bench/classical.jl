using PyCall, Dates

numpy = pyimport("numpy")

include("../src/util.jl")
include("../src/classical/exact.jl")
include("../src/classical/sdp.jl")

function max_test(P, w)
    X_max = max_sdp(var(P, w))
    sdp_max = best_hr(var(P, w), X_max, 3000, true)
    _, exact_max = max_qp(var(P, w))
    return sdp_max, exact_max
end

function fair_test(P, w)
    X_fair = min_max_expected(varl(P, w); maxmin=true)
    sdp_fair = sdp_minmax(varl(P, w), X_fair; maxmin=true)
    _, exact_fair = minmax_expectation(varl(P, w); maxmin=true)
    return sdp_fair, exact_fair
end

function run_tests()
    open("bench/logs/" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_classical.csv", "a") do log
        write(log, "f, sdp max, exact max, sdp fair, exact fair\n")
        for dataset in ["contact-high-school", "email-Enron"]
            for f in readdir("data/$dataset/numpy/")
                P = Matrix(numpy.load("data/$dataset/numpy/$f"))
                w = ones(size(P, 2))
                sdp_max, exact_max = sdp_test(P, w)
                sdp_fair, exact_fair = fair_test(P, w)
                write(log, "$f, $sdp_max, $exact_max, $sdp_fair, $exact_fair\n")
            end
        end
    end
end
