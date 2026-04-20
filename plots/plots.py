import matplotlib.pyplot as plt
import pandas as pd

sqr = (3.5, 2.5)
rec = (7.16, 2.5)

if __name__ == '__main__':
    classicalreal = pd.read_csv("bench/logs/2026-04-20_02-25-06_classical_15.csv")
    quantumreal = pd.read_csv("bench/logs/2026-04-14_11-02-57_quantum.csv")
    real = pd.merge(classicalreal, quantumreal, on="f")
    real["cat"] = real["f"].map(lambda x: x.split("_")[0])
    real["id"] = real["f"].map(lambda x: x.split("_")[1])
    real["SDP Ratio"] = real["ratio fair"]
    real["Quantum Ratio"] = real["quantum fair p=3"] / real["exact fair"]
    real_min = min(real["SDP Ratio"].min(), real["Quantum Ratio"].min())
    greal = real.groupby("cat")

    classicalkarloff = pd.read_csv("bench/logs/2026-04-17_15-12-53_karloff_20.csv")
    karloff = classicalkarloff
    karloff["SDP Ratio"] = karloff["ratio fair"]

    fig, axs = plt.subplots(figsize=rec, 
                        nrows=1, ncols=2,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    targets = zip(greal.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        ax.set_xlabel("SDP Ratio")
        ax.set_title(key)
        g = greal.get_group(key)
        ax.scatter(g["SDP Ratio"], g["Quantum Ratio"])
        ax.plot([real_min, 1], [real_min, 1])

    axs.flatten()[0].set_ylabel("Quantum Ratio")
    # ax.legend()
    plt.savefig("plots/fair.pdf")

    fig, ax = plt.subplots(figsize=sqr, 
                        nrows=1, ncols=1,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    ax.plot([1, 2, 3, 4], [real.iloc[7]["SDP Ratio"]] * 4)
    ax.set_xlabel("QAOA Layers")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("contact-high-school 08")
    plt.savefig("plots/fair_layers.pdf")

    fig, ax = plt.subplots(figsize=sqr, 
                        nrows=1, ncols=1,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    ax.plot([1, 2, 3, 4], [karloff.iloc[3]["SDP Ratio"]] * 4)
    ax.set_xlabel("QAOA Layers")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Karloff")
    plt.savefig("plots/karloff_layers_fair.pdf")


    karloff["SDP Ratio"] = karloff["ratio max"]

    fig, ax = plt.subplots(figsize=sqr, 
                        nrows=1, ncols=1,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    ax.plot([1, 2, 3, 4], [karloff.iloc[3]["SDP Ratio"]] * 4)
    ax.set_xlabel("QAOA Layers")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Karloff")
    plt.savefig("plots/karloff_layers_max.pdf")


    classicalpareto = pd.read_csv("bench/logs/2026-04-20_03-46-56_6_3_1_3.csv")
    pareto = classicalpareto
    pareto["ID"] = 1
    gpareto = pareto.groupby("ID")

    fig, axs = plt.subplots(figsize=rec, 
                        nrows=1, ncols=2,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    targets = zip([1, 1], axs.flatten())
    for i, (key, ax) in enumerate(targets):
        ax.set_xlabel("Variance")
        ax.set_title(key)
        g = gpareto.get_group(key)
        ax.plot(g["sdp 1"], g["sdp 2"], label="SDP")
        ax.plot(g["exact 1"], g["exact 2"], label="Exact")

    axs.flatten()[0].set_ylabel("Imbalance")
    ax.legend()
    plt.savefig("plots/pareto.pdf")

    # fig, axs = plt.subplots(figsize=(9,3), 
    #                     nrows=1, ncols=3,     # fix as above
    #                     gridspec_kw=dict(hspace=0.4)
    #                     , sharey=True) # Much control of gridspec
    # fig.subplots_adjust(bottom=0.15)

    # targets = zip(series.groups.keys(), axs.flatten())
    # for i, (key, ax) in enumerate(targets):
    #     ax.set_xlabel("|V|")
    #     ax.set_yscale("log")
    #     ax.set_title('k=%d'%key)
    #     for j in ["DP", "Gurobi"]:
    #         g = series.get_group(key)
    #         ax.plot(g["n"], g[j], label=j, linewidth=4)

    # axs.flatten()[0].set_ylabel("Seconds")
    # ax.legend()
    # plt.savefig("figs/series.pdf")

    # real.set_index("Graph").plot(kind="barh", width=0.75)
    # fig = plt.gcf()
    # ax = plt.gca()
    
    # fig.subplots_adjust(left=0.30)
    # ax.set_xlabel("Seconds")

    # # axs.flatten()[0].set_ylabel("Seconds")
    # # ax.legend()
    # plt.savefig("figs/real.pdf")
