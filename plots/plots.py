import matplotlib.pyplot as plt
import pandas as pd

sqr = (3.5, 2.5)
rec = (7.16, 2.5)
font = {'family' : 'sans-serif',
        'size'   : 10}
plt.rc('font', **font)

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
        ax.plot([real_min, 1], [real_min, 1], color="red", linestyle="--")

    axs.flatten()[0].set_ylabel("Quantum Ratio")
    # ax.legend()
    plt.savefig("plots/fair.pdf")

    fig, ax = plt.subplots(figsize=(3.5, 3.5), 
                        nrows=1, ncols=1) # Much control of gridspec
    fig.subplots_adjust(left=0.2, bottom=0.2)

    ax.plot([1, 2, 3], [1 - 0.912] * 3, label="SDP")
    ax.plot([1, 2, 3], [1 - 1] * 3, label="Exact")
    ax.plot([1, 2, 3], [1 - 0.813, 1 - 0.848, 1 - 0.99], label="Quantum")
    ax.set_xlabel("QAOA Layers")
    ax.set_ylabel("Greatest Exp. Imbalance")
    ax.legend()
    ax.set_xticks([1, 2, 3])
    plt.savefig("plots/karloff_layers_fair.pdf")

    fig, ax2 = plt.subplots(figsize=(3.5, 3.5), 
                        nrows=1, ncols=1) # Much control of gridspec
    fig.subplots_adjust(left=0.2, bottom=0.2)

    ax2.plot([1, 2, 3], [27.36] * 3, label="SDP")
    ax2.plot([1, 2, 3], [30] * 3, label="Exact")
    ax2.plot([1, 2, 3], [25.46, 28.49, 29.99], label="Quantum")
    ax2.set_xlabel("QAOA Layers")
    ax2.set_ylabel("Total Variance")
    ax2.legend()
    ax2.set_xticks([1, 2, 3])
    plt.savefig("plots/karloff_layers_max.pdf")


    classicalpareto = pd.read_csv("bench/logs/2026-04-21_13-09-40_email-Enron_5.csv")
    pareto = classicalpareto

    fig, (ax1, ax2) = plt.subplots(figsize=rec, 
                        nrows=1, ncols=2,     # fix as above
                        gridspec_kw=dict(hspace=0.4)
                        , sharey=True) # Much control of gridspec
    fig.subplots_adjust(bottom=0.2)

    ax1.set_xlabel("Variance")
    ax1.set_title(key)
    ax1.plot(pareto["sdp 1"], pareto["sdp 2"], label="SDP")
    ax1.plot(pareto["exact 1"], pareto["exact 2"], label="Exact")

    ax1.set_ylabel("Imbalance")
    ax1.legend()

    classicalpareto = pd.read_csv("bench/logs/2026-04-21_13-15-17_email-Enron_5.csv")
    pareto = classicalpareto


    ax2.set_xlabel("Variance")
    ax2.set_title(key)
    ax2.plot(pareto["sdp 1"], pareto["sdp 2"], label="SDP")
    ax2.plot(pareto["exact 1"], pareto["exact 2"], label="Exact")

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
