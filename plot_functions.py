
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_simple(r1, r2):

    mpl.rcdefaults()
    plt.style.use("default")
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6), dpi=300)

    for a, r in [(ax[0], r1), (ax[1], r2)]:

        a.plot(r.alphas, r.xi0s, color="red", label=r"$-\xi$", zorder=np.floor(r.xi0_alpha*1e3))
        if r.xi0_alpha is not None:
            a.plot(r.xi0_alpha, 1, marker="*", color="red", markersize=8, zorder=np.floor(r.xi0_alpha*1e3))

        a.plot(r.alphas, r.xi1s, color="orange", label=r"$-\xi_1$", zorder=np.floor(r.xi1_alpha*1e3))
        if r.xi1_alpha is not None:
            a.plot(r.xi1_alpha, 1, marker="*", color="orange", markersize=8, zorder=np.floor(r.xi1_alpha*1e3))

        a.plot(r.alphas, r.rho0s, color="blue", label=r"$\rho$", zorder=np.floor(r.rho0_alpha*1e3))
        if r.rho0_alpha is not None:
            a.plot(r.rho0_alpha, 1, marker="*", color="blue", markersize=8, zorder=np.floor(r.rho0_alpha*1e3))

        a.plot(r.alphas, r.rho1s, color="green", label=r"$\rho_1$", zorder=np.floor(r.rho1_alpha*1e3))
        if r.rho1_alpha is not None:
            a.plot(r.rho1_alpha, 1, marker="*", color="green", markersize=8, zorder=np.floor(r.rho1_alpha*1e3))

        a.plot(r.alphas, r.rho2s, color="purple", label=r"$\rho_2$", zorder=np.floor(r.rho2_alpha*1e3))
        if r.rho2_alpha is not None:
            a.plot(r.rho2_alpha, 1, marker="*", color="purple", markersize=8, zorder=np.floor(r.rho2_alpha*1e3))

        if r.J_alpha is not None:
            a.axvline(r.J_alpha, ls="--", c="k", zorder=np.floor(r.J_alpha*1e3))

        if r.neg_weight_alpha is not None:
            a.axvline(r.neg_weight_alpha, ls="--", c="0.5", zorder=np.floor(r.neg_weight_alpha*1e3))

        a.axhline(1, c="k", linewidth=1)

        a.grid(True, alpha=0.3)
        a.set_xlabel(r"$\alpha$", fontsize=12)
        ymax = 2
        a.set_ylim(-ymax / 10, ymax)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.1),
        handlelength=1.5
    )

    fig.tight_layout()
    fig.savefig("simple.pdf", bbox_inches="tight")
    plt.show()

def plot_second_order(r1, r2):

    mpl.rcdefaults()
    plt.style.use("default")
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6), dpi=300)

    for a, r in [(ax[0], r1), (ax[1], r2)]:

        if r.J_alpha is not None:
            a.axvline(x=r.J_alpha, ls="--", color="k", zorder=np.floor(r.J_alpha*1e3))

        if r.skar_alpha is not None:
            a.axvline(x=r.skar_alpha, ls="--", color="grey", label=r"$\lambda_2=0$")

        a.plot(r.alphas, r.phi_L, color="red", label=r"$\overline{\phi}(\boldsymbol{L})$", zorder=np.floor(r.phi_L_alpha*1e3))
        if r.phi_L_alpha is not None:
            y = r.phi_L[r.alphas.index(r.phi_L_alpha)]
            a.plot(r.phi_L_alpha, y, marker="*", color="red", markersize=8, zorder=np.floor(r.phi_L_alpha*1e3))

        a.plot(r.alphas, r.phi_rho0, color="blue", label=r"$\arcsin(\rho)$", zorder=np.floor(r.phi_rho0_alpha*1e3))
        if r.phi_rho0_alpha is not None:
            y = r.phi_rho0[r.alphas.index(r.phi_rho0_alpha)]
            a.plot(r.phi_rho0_alpha, y, marker="*", color="blue", markersize=8, zorder=np.floor(r.phi_rho0_alpha*1e3))

        a.plot(r.alphas, r.phi_rho1, color="green", label=r"$\arcsin(\rho_1)$", zorder=np.floor(r.phi_rho1_alpha*1e3))
        if r.phi_rho1_alpha is not None:
            y = r.phi_rho1[r.alphas.index(r.phi_rho1_alpha)]
            a.plot(r.phi_rho1_alpha, y, marker="*", color="green", markersize=8, zorder=np.floor(r.phi_rho1_alpha*1e3))

        a.plot(r.alphas, r.phi_rho2, color="purple", label=r"$\arcsin(\rho_2)$", zorder=np.floor(r.phi_rho2_alpha*1e3))
        if r.phi_rho2_alpha is not None:
            y = r.phi_rho2[r.alphas.index(r.phi_rho2_alpha)]
            a.plot(r.phi_rho2_alpha, y, marker="*", color="purple", markersize=8, zorder=np.floor(r.phi_rho2_alpha*1e3))

        a.set_xlabel(r"$\alpha$", fontsize=12)
        a.grid(True, alpha=0.3)
        a.set_ylim(None, np.pi / 2)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.1),
        handlelength=1.5
    )

    fig.tight_layout()
    fig.savefig("second_order.pdf", bbox_inches="tight")
    plt.show()