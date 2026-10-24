"""Phase 1 du plan — Calibrage 2D à un seul point : fonction plate + une gaussienne.

E1a : cibles tirées dans l'ensemble atteignable {I + a vᵀ : ‖v‖ ≤ 0,7·e^(−1/2)/R₀} (+ variante bruitée) ;
      variantes V1 (x, y, log R), V2 (x, y ; R = R₀ fixé), V3 (+ a libre), baselines LoRA à a fixé
      et matrice 2×2 libre.
E1b : cibles 2×2 quelconques — gap mesuré contre la valeur théorique ‖(I − ââᵀ)(W* − I)‖_F.
Mesures : vitesse de convergence (pas jusqu'à 1e-10), stabilité inter-graines (écart-type des pertes
finales), spectre de la hessienne en V1 (valeur propre nulle attendue : la direction plate).
Sortie : résultats dans results_phase1.json + impressions.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from genjac import phase1_loss_grad, lora_loss_grad, adam

U0 = np.array([0.5, 0.5])          # point d'évaluation unique
AFIX = np.array([0.6, 0.8])        # a fixé et normalisé
R0 = 0.5                           # rayon fixé de la variante V2
VMAX_V2 = np.exp(-0.5) / R0        # ‖v‖ max atteignable en V2 (max de d·φ(d)/R² à d = R)

TOL = 1e-13
NSTEPS = 3000


def inits(variant, rng):
    if variant == "V2":
        return U0 + rng.standard_normal(2) * 0.4
    if variant == "V1":
        return np.concatenate([U0 + rng.standard_normal(2) * 0.4,
                                [rng.uniform(np.log(0.3), np.log(1.5))]])
    return np.concatenate([U0 + rng.standard_normal(2) * 0.4,
                           [rng.uniform(np.log(0.3), np.log(1.5))],
                           rng.standard_normal(2) * 0.5])


def run_variant(variant, T, seed, steps=NSTEPS, tol=TOL):
    rng = np.random.default_rng(seed)
    if variant == "lora":
        x, L, st = adam(lambda w: lora_loss_grad(w, AFIX, T), rng.standard_normal(2) * 0.5,
                        lr=0.05, steps=steps, tol=tol)
    elif variant == "libre":
        # matrice 2×2 libre : minimum exact = T (plafond de référence)
        return float(np.linalg.norm(np.zeros(2) * 0)), 0.0, 0
    else:
        x, L, st = adam(lambda p: phase1_loss_grad(p, U0, AFIX, T, variant, R0=R0),
                        inits(variant, rng), lr=0.05, steps=steps, tol=tol)
    return L, st, x


def e1a(ninst=12, nseed=5, sigma=0.0):
    """Cibles dans l'ensemble atteignable commun (borne 0,7·VMAX_V2)."""
    out = {}
    for lab in ("V1", "V2", "V3", "lora", "libre"):
        finals, steps10, stds, taux, all_losses = [], [], [], [], []
        nruns = 0
        for i in range(ninst):
            rng = np.random.default_rng(1000 + i)
            v = rng.standard_normal(2)
            v = 0.7 * VMAX_V2 * v / np.linalg.norm(v) * rng.uniform(0.2, 1.0)
            T = np.eye(2) + np.outer(AFIX, v)
            if sigma:
                T = T + sigma * rng.standard_normal((2, 2))
            fl, st = [], []
            for s in range(nseed):
                L, stp, x = run_variant(lab, T, 2000 + 100 * i + s, steps=2500, tol=0.0 if sigma else TOL)
                fl.append(L); st.append(stp)
                all_losses.append(L)
                taux.append(1 if L < 1e-10 else 0)   # on compte aussi les échecs
            stds.append(np.std(fl))
            finals.append(np.median(fl)); steps10.append(np.median(st))
        out[lab] = {"perte_finale_mediane": float(np.median(finals)),
                    "pas_median": float(np.median(steps10)),
                    "std_inter_graines_median": float(np.median(stds)),
                    "perte_max_run": float(max(all_losses)),
                    "taux_convergence_1e-10": (float(np.mean(taux)) if taux else 0.0)}
        print(f"  E1a{' bruit σ=' + str(sigma) if sigma else '':14s} {lab:5s} perte(méd) "
              f"{out[lab]['perte_finale_mediane']:.3e}  pas(méd) {out[lab]['pas_median']:6.0f}  "
              f"std graines(méd) {out[lab]['std_inter_graines_median']:.2e}  "
              f"conv<1e-10 {out[lab]['taux_convergence_1e-10']:.0%}")
    return out


def e1b(ninst=12, nseed=5):
    """Cibles quelconques : gap mesuré vs théorique."""
    out = {}
    P = np.eye(2) - np.outer(AFIX, AFIX)
    for lab in ("V1", "V2", "V3", "lora", "libre"):
        gaps, excess = [], []
        for i in range(ninst):
            rng = np.random.default_rng(3000 + i)
            G = 0.8 * rng.standard_normal((2, 2))
            T = np.eye(2) + G
            gap_th = float(np.sqrt(np.sum((P @ G) ** 2)))
            best = np.inf
            for s in range(nseed):
                L, stp, x = run_variant(lab, T, 4000 + 100 * i + s, steps=2500, tol=0.0)
                best = min(best, np.sqrt(2 * L))
            gaps.append(best)
            excess.append(best - gap_th)
        out[lab] = {"gap_median": float(np.median(gaps)),
                    "exces_median": float(np.median(excess)),
                    "exces_max": float(np.max(excess))}
        print(f"  E1b {lab:5s} gap(méd) {out[lab]['gap_median']:.4f}  excès(méd) "
              f"{out[lab]['exces_median']:.4f}  excès(max) {out[lab]['exces_max']:.4f}")
    # gap théorique médian pour référence
    ths = []
    for i in range(ninst):
        rng = np.random.default_rng(3000 + i)
        G = 0.8 * rng.standard_normal((2, 2))
        ths.append(float(np.sqrt(np.sum((P @ G) ** 2))))
    out["gap_theorique_median"] = float(np.median(ths))
    print(f"  E1b gap théorique (méd) : {out['gap_theorique_median']:.4f}")
    return out


def hessianes_v1(ninst=5):
    """Spectre de la hessienne de la perte V1 au point convergé : valeur propre ~0 attendue
    (la direction plate (d, R) qui laisse v invariant)."""
    eigs = []
    for i in range(ninst):
        rng = np.random.default_rng(5000 + i)
        v = rng.standard_normal(2); v = 0.5 * VMAX_V2 * v / np.linalg.norm(v)
        T = np.eye(2) + np.outer(AFIX, v)
        x, L, st = adam(lambda p: phase1_loss_grad(p, U0, AFIX, T, "V1", R0=R0),
                        inits("V1", rng), lr=0.05, steps=NSTEPS, tol=TOL)
        h = 1e-5
        H = np.zeros((3, 3))
        for j in range(3):
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            H[:, j] = (phase1_loss_grad(xp, U0, AFIX, T, "V1", R0=R0)[1]
                       - phase1_loss_grad(xm, U0, AFIX, T, "V1", R0=R0)[1]) / (2 * h)
        H = 0.5 * (H + H.T)
        w = np.sort(np.linalg.eigvalsh(H))
        eigs.append(list(w))
        print(f"  Hessienne V1 #{i} : valeurs propres {np.array2string(w, precision=6)} "
              f"(min/max = {w[0] / max(w[-1], 1e-300):.2e})")
    minrel = min(w[0] / max(w[-1], 1e-300) for w in eigs)
    return {"spectres": eigs, "rapport_min_max": float(minrel)}


if __name__ == "__main__":
    res = {}
    print("=== E1a — cibles dans l'ensemble atteignable ===")
    res["E1a"] = e1a()
    print("=== E1a (bruit σ = 0,01) ===")
    res["E1a_bruit"] = e1a(sigma=0.01)
    print("=== E1b — cibles quelconques, gap vs théorique ===")
    res["E1b"] = e1b()
    print("=== Hessienne V1 — direction plate ===")
    res["hessienne_v1"] = hessianes_v1()
    with open(Path(__file__).parents[1] / "results" / "results_phase1.json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print("\nRésultats sauvés dans results/results_phase1.json")
