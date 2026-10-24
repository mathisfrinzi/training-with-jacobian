# -*- coding: utf-8 -*-
"""Figure Phase 2 : RMSE de généralisation RELATIVE à la baseline triviale « W = I » (1 = aussi bon que
prédire l'identité ; < 1 = mieux). Familles (a) intégrable dans la famille gaussienne, (mis) intégrable hors famille."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "results_phase2.json").read_text())
names = [("gen", "générateur jacobien", "#d62728"), ("gaussfree", "même base, sans intégrabilité", "#ff7f0e"),
         ("hyp", "hypernetwork", "#1f77b4"), ("hyp_res", "hypernetwork résiduel", "#17becf"),
         ("const_mean", "moyenne des matrices d'entraînement", "#7f7f7f")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, fam, title in zip(axes, ["a", "mis"], ["(a) cible intégrable dans la famille gaussienne",
                                               "(mis) cible intégrable hors famille (tanh)"]):
    cfgs = [c for c in d["configs"] if fam in c["families"] and "controls" in c["families"][fam]]
    x = np.arange(len(cfgs))
    w = 0.16
    for i, (k, lab, col) in enumerate(names):
        y = [c["families"][fam]["controls"][k]["med"] / c["families"][fam]["controls"]["identity"]["med"] for c in cfgs]
        ax.bar(x + (i - 2) * w, y, w, label=lab, color=col)
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"d={c['d']}, m={c['m']}" for c in cfgs])
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("RMSE de généralisation / RMSE(W = I)  (médiane, 8 graines)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase2_controles.svg")
print("→ figures/phase2_controles.svg")
