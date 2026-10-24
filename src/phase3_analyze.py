# -*- coding: utf-8 -*-
"""Analyse de la Phase 3 (transformer 28 k) — results/phase3/.

Protocole :
  - appariement RÉEL par graine : tous les fournisseurs voient la même initialisation du modèle et la
    même séquence de lots (rng de lots séparé) ;
  - tests d'ÉQUIVALENCE (TOST, IC 90 % du Δ apparié dans ±0,005 nats) au lieu de « p grand = égalité » ;
  - ALBERT et dense présents au même nombre de pas ; ablation `genfree` (non intégrable) ;
  - diagnostics : activité des gaussiennes (φ_max), distances inter-couches, rangs effectifs,
    perte sur validation bruitée ;
  - condition préalable de DISCRIMINATION : dense doit battre ALBERT, sinon la tâche ne voit pas la
    paramétrisation de W₁.
Sortie : results/phase3/summary.json et figures/phase3_courbes.svg
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase3"
FIG = ROOT / "figures"
METHODS = ["gen", "genfree", "lora", "hyp", "albert", "dense", "gen_naive"]
LABEL = {"gen": "Générateur (init vivante)", "genfree": "Ablation non intégrable", "lora": "LoRA ×4",
         "hyp": "Hypernetwork", "albert": "ALBERT", "dense": "Dense", "gen_naive": "Générateur (init naïve)"}
MARGIN = 0.005


def load(task="a"):
    out = {}
    for m in METHODS:
        for f in sorted(RES.glob(f"{task}_{m}_s*.json")):
            if f.name.endswith("_W1.json"):
                continue
            r = json.loads(f.read_text())
            if r.get("protocol") == "final" and r["steps"] == 4000:
                out.setdefault(m, {})[r["seed"]] = r
    return out


def paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = mean / se if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), n - 1))
    tc = stats.t.ppf(0.95, n - 1)                       # IC 90 % bilatéral = TOST à 5 %
    lo, hi = mean - tc * se, mean + tc * se
    return {"n": n, "delta": float(mean), "p": p, "ci90": [float(lo), float(hi)],
            "equivalent": bool(lo > -MARGIN and hi < MARGIN)}


def main():
    data = load("a")
    seeds_all = sorted(set.intersection(*[set(v) for v in data.values()])) if data else []
    summary = {"seeds": seeds_all, "methods": {}, "paired_vs_gen": {}, "paired_other": {}}
    print(f"graines communes : {seeds_all}")
    for m, runs in data.items():
        v = np.array([runs[s]["final_val"] for s in seeds_all])
        vn = np.array([runs[s]["final_val_noise"] for s in seeds_all])
        row = {"val_mean": float(v.mean()), "val_std": float(v.std(ddof=1)), "val_noise_mean": float(vn.mean()),
               "ff1_params": runs[seeds_all[0]]["ff1_params"], "vals": v.tolist(),
               "mean_interlayer_dist": float(np.mean([runs[s]["w1_diag"]["mean_dist"] for s in seeds_all])),
               "eff_rank": float(np.mean([np.mean(runs[s]["w1_diag"]["eff_ranks"]) for s in seeds_all]))}
        if m in ("gen", "gen_naive"):
            row["phi_max_final"] = float(np.mean([runs[s]["gen_diag"]["phi_max"] for s in seeds_all]))
            row["n_active_final"] = float(np.mean([runs[s]["gen_diag"]["n_active_gt_0.01"] for s in seeds_all]))
        summary["methods"][m] = row
        print(f"{LABEL[m]:30s} val {v.mean():.4f} ± {v.std(ddof=1):.4f}  bruit {vn.mean():.3f}  "
              f"W1={row['ff1_params']}  dist.inter-couches={row['mean_interlayer_dist']:.3f}"
              + (f"  φmax={row['phi_max_final']:.2f}" if "phi_max_final" in row else ""))
    V = {m: [data[m][s]["final_val"] for s in seeds_all] for m in data}
    for other in ["lora", "hyp", "albert", "dense", "genfree", "gen_naive"]:
        if "gen" in V and other in V:
            summary["paired_vs_gen"][other] = paired(V["gen"], V[other])
            r = summary["paired_vs_gen"][other]
            print(f"Δ gen−{other:10s}: {r['delta']:+.4f}  p={r['p']:.3f}  IC90=[{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]"
                  f"  {'ÉQUIVALENT' if r['equivalent'] else ''}")
    if "dense" in V and "albert" in V:
        summary["paired_other"]["dense_vs_albert"] = paired(V["dense"], V["albert"])
        r = summary["paired_other"]["dense_vs_albert"]
        discr = r["delta"] < -MARGIN and r["p"] < 0.05
        summary["discrimine"] = bool(discr)
        print(f"Discrimination : Δ(dense−albert) = {r['delta']:+.4f} (p={r['p']:.3f}) → "
              f"{'tâche discriminante' if discr else 'TÂCHE NON DISCRIMINANTE'}")
    # stabilité : test F de variance (gen vs lora/hyp)
    summary["variance_tests"] = {}
    for other in ["lora", "hyp"]:
        if "gen" in V and other in V:
            f = np.var(V[other], ddof=1) / np.var(V["gen"], ddof=1)
            n = len(seeds_all) - 1
            summary["variance_tests"][other] = {"F": float(f), "p_two_sided": float(min(1.0, 2 * min(stats.f.cdf(f, n, n), stats.f.sf(f, n, n))))}
            print(f"Stabilité σ²({other})/σ²(gen) = {f:.2f}  (p bilatéral {summary['variance_tests'][other]['p_two_sided']:.2f})")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for m in data:
            steps = [c["step"] for c in data[m][seeds_all[0]]["curve"]]
            Y = np.array([[c["val"] for c in data[m][s]["curve"]] for s in seeds_all])
            ax.plot(steps, Y.mean(0), label=LABEL[m], lw=2.0 if m == "gen" else 1.2,
                    ls="--" if m in ("gen_naive", "genfree") else "-")
        ax.set_ylim(0.98, 1.4)
        ax.set_xlabel("Pas"); ax.set_ylabel("Perte de validation (moyenne sur graines)")
        ax.set_title("Phase 3 — tâche A, 4 000 pas"); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        fig.savefig(FIG / "phase3_courbes.svg")
        print("→ figures/phase3_courbes.svg")
    except Exception as e:                                      # pragma: no cover
        print("figure sautée :", e)


if __name__ == "__main__":
    main()
