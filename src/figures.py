"""Figure de la Phase 1 : (a) perte le long des directions propres de la hessienne V1 au point convergé
(la direction plate ~ nulle vs les autres), (b) spectre des 5 instances.
Sortie : figures/phase1_direction_plate.svg"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genjac import phase1_loss_grad, adam
from phase1 import U0, AFIX, R0, VMAX_V2, inits

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)

C_GEN, C_HYP = "#2563eb", "#ef4444"


# --------------------------------------------------------------------- figure 1
def fig_phase1():
    rng = np.random.default_rng(5000)
    v = rng.standard_normal(2)
    v = 0.5 * VMAX_V2 * v / np.linalg.norm(v)
    T = np.eye(2) + np.outer(AFIX, v)
    x, L, st = adam(lambda p: phase1_loss_grad(p, U0, AFIX, T, "V1", R0=R0),
                    inits("V1", rng), lr=0.05, steps=3000, tol=1e-13)
    # hessienne par différences finies
    h = 1e-5
    H = np.zeros((3, 3))
    for j in range(3):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        H[:, j] = (phase1_loss_grad(xp, U0, AFIX, T, "V1", R0=R0)[1]
                   - phase1_loss_grad(xm, U0, AFIX, T, "V1", R0=R0)[1]) / (2 * h)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    ts = np.linspace(-0.6, 0.6, 120)
    curves = []
    for j in range(3):
        Ls = [phase1_loss_grad(x + t * V[:, j], U0, AFIX, T, "V1", R0=R0)[0] for t in ts]
        curves.append(np.array(Ls))

    data = json.loads((RES / "results_phase1.json").read_text())
    specs = np.array(data["hessienne_v1"]["spectres"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    for j, (col, lab) in enumerate(zip(["#16a34a", "#f59e0b", "#dc2626"],
                                       [f"direction plate (λ = {w[0]:.1e})",
                                        f"λ₂ = {w[1]:.2f}", f"λ₃ = {w[2]:.2f}"])):
        ax.semilogy(ts, curves[j] + 1e-30, color=col, label=lab, lw=2)
    ax.set_xlabel("déplacement t le long de la direction propre")
    ax.set_ylabel("perte L (échelle log)")
    ax.set_title("(a) Perte V1 au point convergé\nla direction (c, log R) est plate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, s in enumerate(specs):
        ax.scatter(range(3), np.maximum(s, 1e-11), color=C_GEN, zorder=3,
                   label="instances V1" if i == 0 else None)
    ax.axhline(1e-11, color="k", ls=":", lw=1)
    ax.text(0.05, 1.4e-11, "précision numérique", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["λ₁ (min)", "λ₂", "λ₃"], fontsize=9)
    ax.set_ylabel("valeur propre")
    ax.set_title("(b) Spectre de la hessienne V1 (5 instances)\n"
                 f"rapport λ₁/λ₃ min = {data['hessienne_v1']['rapport_min_max']:.1e}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "phase1_direction_plate.svg")
    print("figure 1 ok")


if __name__ == "__main__":
    fig_phase1()
