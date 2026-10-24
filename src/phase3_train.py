"""Phase 3 — Entraîneur : un transformer dont la matrice W1 du FFN vient d'un fournisseur.

CLI : python src/phase3_train.py --task a|b --method gen|lora|hyp|albert|dense --seed 0 [--steps N]
Sauvegarde incrémentale : results/phase3/{task}_{method}_s{seed}.json + matrices W1 en .npz.
Protocole verrouillé (pré-enregistré dans PLAN.md) : budget total apparié sur W1, mêmes
hyperparamètres pour toutes les méthodes, courbe d'apprentissage sur perte de validation,
mesures finales = val + val bruitée (10 % de caractères permutés) + spectre singulier des Wℓ.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_model import Transformer
from phase3_methods import make_provider
from phase3_corpus import get_task

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

# --- protocole verrouillé ---
D, D_FF, L, H = 32, 64, 4, 2
T, B = 64, 16
STEPS = 1500
LR0, LR1 = 3e-3, 3e-4          # décroissance cosinus
EVAL_EVERY = 100
N_EVAL = 64                     # fenêtres de validation


def cosine_lr(step, total):
    return LR1 + 0.5 * (LR0 - LR1) * (1 + np.cos(np.pi * step / total))


def w1_diagnostics(Ws):
    """Distances relatives inter-couches ‖Wℓ − W₀‖/‖W₀‖ et rangs effectifs (exp de l'entropie spectrale)."""
    Ws = [np.asarray(W, dtype=np.float64) for W in Ws]
    dist = [float(np.linalg.norm(W - Ws[0]) / np.linalg.norm(Ws[0])) for W in Ws[1:]]
    ranks = []
    for W in Ws:
        sv = np.linalg.svd(W, compute_uv=False)
        p = sv / sv.sum()
        ranks.append(float(np.exp(-(p * np.log(p + 1e-30)).sum())))
    return {"dist_to_layer0": dist, "mean_dist": float(np.mean(dist)), "eff_ranks": ranks}


def gen_diagnostics(prov):
    """Activité des gaussiennes du générateur : φ_max(ℓ,k) = exp(−‖uℓ−cₖ‖²/2Rₖ²) et déviation à M."""
    p = prov.opt.params
    R = np.exp(p["logR"].astype(np.float64))
    U, C = p["U"].astype(np.float64), p["C"].astype(np.float64)
    phi = np.array([[np.exp(-np.sum((U[l] - C[k]) ** 2) / (2 * R[k] ** 2)) for k in range(C.shape[0])]
                    for l in range(U.shape[0])])
    return {"phi_max": float(phi.max()), "phi_median": float(np.median(phi)),
            "n_active_gt_0.01": int((phi > 0.01).sum())}


def run(task, method, seed, steps=STEPS, out=None):
    """RNG séparés : `rng` initialise modèle puis fournisseur ; `brng` ne tire QUE les
    lots d'entraînement, avec la même graine pour toutes les méthodes ⇒ vrai appariement par graine."""
    out = Path(out) if out is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, val = get_task(task, seed=0)          # corpus identiques pour toutes les méthodes
    val_noise = val.perturbed(0.1, seed=seed + 777)
    rng = np.random.default_rng(1000 + seed)
    brng = np.random.default_rng(50000 + seed)
    model = Transformer(train.V, D, D_FF, L, H, T, rng)
    prov = make_provider(method, D, D_FF, L, rng, K=6, r=2, e_dim=8, Hh=6)
    Xval = val.eval_windows(T, N_EVAL)
    Xval_n = val_noise.eval_windows(T, N_EVAL)

    curve = []
    for step in range(steps):
        X = train.sample_batch(B, T, brng)
        W1s = prov.W_layers()
        loss, cache = model.forward(X, W1s)
        grads, Gs = model.backward(cache, W1s)
        prov.backprop(Gs)
        lr = cosine_lr(step, steps)
        model.step(grads, lr)
        prov.step(lr)
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            W1s = prov.W_layers()
            curve.append({"step": step + 1, "train": loss,
                          "val": model.eval_loss(Xval, W1s)})

    W1s = prov.W_layers()
    final_val = model.eval_loss(Xval, W1s)
    final_train = model.eval_loss(train.eval_windows(T, N_EVAL, seed=999), W1s)
    final_noise = model.eval_loss(Xval_n, W1s)

    res = {
        "task": task, "method": method, "seed": seed, "steps": steps,
        "ff1_params": prov.n_params,
        "model_params": int(sum(v.size for v in model.params.values())),
        "final_train": final_train, "final_val": final_val,
        "final_val_noise": final_noise,
        "best_val": min(c["val"] for c in curve),
        "curve": curve, "time_s": round(time.time() - t0, 1),
        "w1_diag": w1_diagnostics(W1s),
        "protocol": "final",
    }
    if method in ("gen", "gen_naive"):
        res["gen_diag"] = gen_diagnostics(prov)
    (out / f"{task}_{method}_s{seed}.json").write_text(json.dumps(res))
    np.savez(out / f"{task}_{method}_s{seed}_W1.npz",
             W=np.stack([np.asarray(W, dtype=np.float32) for W in W1s]))
    print(f"[done] {task}/{method}/s{seed}: val={final_val:.4f} noise={final_noise:.4f} "
          f"({res['time_s']}s, ff1={prov.n_params})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--method", choices=["gen", "gen_naive", "genfree", "lora", "hyp", "albert", "dense"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : results/phase3)")
    args = ap.parse_args()
    run(args.task, args.method, args.seed, args.steps, args.out)
