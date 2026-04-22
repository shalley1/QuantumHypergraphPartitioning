using PyCall, Dates, Graphs, FastChebInterp

numpy = pyimport("numpy")

include("../src/util.jl")
include("../src/classical/exact.jl")
include("../src/classical/sdp.jl")

function max_test(P, w=ones(size(P, 2)))
    X_max = max_sdp(var(P, w))
    sdp_max = dot(var(P, w), expected_hr(X_max))
    # _, sdp_max = best_hr(var(P, w), X_max, 1000, true)
    _, exact_max = max_qp(var(P, w))
    return r6(sdp_max), r6(exact_max)
end

function fair_test(P, w=ones(size(P, 2)))
    X_fair = min_max_expected(varl(P, w); maxmin=true)
    sdp_fair = sdp_minmax(varl(P, w), X_fair; maxmin=true)
    _, exact_fair = minmax_expectation(varl(P, w); maxmin=true)
    return r6(sdp_fair), r6(exact_fair)
end

r6(x) = round(x; sigdigits=6)

function pareto_test(P1, P2, w1, w2, α)
    M = imb(P1, w1)
    V = var(P2, w2)
    O = α .* M ./ eigmax(M) + (1 - α) .* V ./ eigmax(V)
    X_max = max_sdp(O)
    S = expected_hr(X_max)
    exact_x, _ = max_qp(O)
    sdp_pareto1, sdp_pareto2 = r6(dot(M, S)), r6(dot(V, S))
    exact_pareto1, exact_pareto2 = r6(dot(M * exact_x, exact_x)), r6(dot(V * exact_x, exact_x))
    return sdp_pareto1, sdp_pareto2, exact_pareto1, exact_pareto2
end

function run_tests(tag, target)
    open("bench/logs/" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_$(tag)_$target.csv", "a") do log
        write(log, "f,sdp max,exact max,ratio max,sdp fair,exact fair,ratio fair\n")
        for dataset in ["congress-bills"] # ["contact-high-school", "email-Enron"] # ["noisy_karloff"] #
            for f in readdir("data/$dataset/numpy_$target/")
                P = Matrix(numpy.load("data/$dataset/numpy_$target/$f"))
                w = ones(size(P, 2)) # [length(nzrange(sparse(P), i))^2 for i in axes(P, 2)]
                write(log, "$(splitext(f)[1]), ")
                sdp_max, exact_max = max_test(P, w)
                write(log, "$sdp_max,$exact_max,$(sdp_max / exact_max),")
                sdp_fair, exact_fair = fair_test(P, w)
                write(log, "$sdp_fair,$exact_fair,$(sdp_fair / exact_fair)")
                write(log, "\n")
            end
        end
    end
end


function run_pareto(folder, fname)
    open("bench/logs/" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_$fname.csv", "a") do log
        write(log, "α,sdp 1,sdp 2,exact 1,exact 2\n")
        P = Matrix(numpy.load("$folder/$fname.npy"))
        w = ones(size(P, 2)) # [length(nzrange(sparse(P), i))^2 for i in axes(P, 2)]
        P1, w1 = P, w
        P2, w2 = ones(size(P1, 1), 1) ./ size(P1, 1), [1]
        for α in chebpoints(20, 0, 1)
            sdp_pareto1, sdp_pareto2, exact_pareto1, exact_pareto2 = pareto_test(P1, P2, w1, w2, α)
            write(log, "$α,$sdp_pareto1,$sdp_pareto2,$exact_pareto1,$exact_pareto2\n")
        end
    end
end
