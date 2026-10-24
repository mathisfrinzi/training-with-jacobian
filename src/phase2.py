"""Phase 2 du plan — Le vrai test : un même g évalué à m points, généralisation à points non vus.

Configs primaires : m = 4, d ∈ {2, 4, 8} à budgets appariés —
    d=2 : générateur K=4 (20 params) vs hypernet H=2 (18) ;
    d=4 : K=5 (45) vs H=1 (37) ;
    d=8 : K=9 (153) vs H=1 (137).
Sweep : m ∈ {2, 8} à d = 2.

Familles de cibles :
    a  : intégrables — jacobiennes d'un g* connu à K*=3 gaussiennes (train + 100 points non vus) ;
    b1 : matrices indépendantes T_i = I + 0,5·N(0,1) (train seulement, rien à généraliser) ;
    b2 : champ lisse non intégrable W*(u)_jk = δ_jk + 0,5·sin(w_jᵀu + b_k) (train + non vus).

Méthodes : générateur jacobien, hypernetwork à sortie libre (baseline clé), m matrices libres
(fit exact), m LoRA indépendantes (rang 1 par point, résolues en forme fermée par SVD).
Métriques : RMSE train = sqrt(2L/m), RMSE généralisation, écart d'intégrabilité (curl) par
différences finies aux points non vus, stabilité inter-graines (8 graines instance+init).

Sortie : fusion progressive dans results/results_phase2.json.

Contrôles et baselines :
  - baselines triviales : « W = I » et « moyenne des matrices d'entraînement » ;
  - hypernetwork « résiduel » hyp_res (W = I + h(u), même budget) = baseline à prior égal, et hyp_big
    (H = 8, hors budget) = borne de capacité ;
  - ablation « même base gaussienne SANS contrainte d'intégrabilité » (W = I + Σ φₖ(u) aₖbₖᵀ) : isole
    l'apport de l'intégrabilité ;
  - famille « mis » : cibles intégrables HORS famille gaussienne (g*(u) = u + ½·B·tanh(Cu + e)) ;
  - valeurs par graine conservées + test de Wilcoxon apparié + nombre de victoires.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genjac import loss_and_grad, W_field, nparams, adam
from hypernet import Hypernet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "results_phase2.json"

NSTEPS = 4000
TOL = 1e-12
LR = 0.05
NTEST = 100
NSEEDS = 8
KSTAR = 3            # nombre de gaussiennes du g* cible (famille a)
UBOX = 1.5           # points u ~ U([-1.5, 1.5]^d)
NCURL = 25           # points non vus utilisés pour le curl (sous-ensemble)

BUDGET = {2: dict(K=4, H=2), 4: dict(K=5, H=1), 8: dict(K=9, H=1)}
FAM_ID = {"a": 0, "b1": 1, "b2": 2, "mis": 9}
KGF = {2: 3, 4: 3, 8: 6}      # gaussiennes de l'ablation libre (budget K(3d+1) ≈ celui du générateur)
HBIG = 8


# -------------------------------------------------------------------------------- cibles

def make_target(family, d, m, rng):
    """Retourne (u_tr, T_tr, u_te, T_te) ; T_te = None pour la famille b1."""
    u_tr = rng.uniform(-UBOX, UBOX, size=(m, d))
    u_te = rng.uniform(-UBOX, UBOX, size=(NTEST, d))
    if family == "a":
        # g* à K*=3 gaussiennes : c* ~ 0,8·N(0,1), logR* ~ U(log 0,4 ; log 1,6), a* ~ 0,5·N(0,1)
        c = rng.standard_normal((KSTAR, d)) * 0.8
        logR = rng.uniform(np.log(0.4), np.log(1.6), KSTAR)
        a = rng.standard_normal((KSTAR, d)) * 0.5
        theta = np.concatenate([c.reshape(-1), logR, a.reshape(-1)])
        return u_tr, W_field(theta, u_tr, KSTAR, d), u_te, W_field(theta, u_te, KSTAR, d)
    if family == "mis":
        # Intégrable mais HORS famille gaussienne : g*(u) = u + ½·B·tanh(Cu + e) ⇒ J = I + ½ B diag(1−tanh²) C
        h = 3
        B_ = rng.standard_normal((d, h)) / np.sqrt(h)
        C_ = rng.standard_normal((h, d)) / np.sqrt(d) * 1.5
        e_ = rng.standard_normal(h) * 0.5

        def J(u):
            t = np.tanh(u @ C_.T + e_)
            return np.eye(d)[None] + 0.5 * np.einsum("ih,mh,hj->mij", B_, 1 - t ** 2, C_)

        return u_tr, J(u_tr), u_te, J(u_te)
    if family == "b1":
        T_tr = np.tile(np.eye(d), (m, 1, 1)) + 0.5 * rng.standard_normal((m, d, d))
        return u_tr, T_tr, u_te, None
    if family == "b2":
        w = rng.standard_normal((d, d)) * 0.5          # lignes w_j
        b = rng.standard_normal(d) * 0.5                # b_k

        def field(u):
            S = u @ w.T                                 # (m,d) : S[i,j] = w_j·u_i
            return np.eye(d)[None, :, :] + 0.5 * np.sin(S[:, :, None] + b[None, None, :])

        return u_tr, field(u_tr), u_te, field(u_te)
    raise ValueError(family)


# -------------------------------------------------------------------------------- méthodes

def fit_generator(u_tr, T_tr, K, d, rng, steps=NSTEPS):
    c0 = rng.standard_normal((K, d))
    logR0 = rng.uniform(np.log(0.3), np.log(1.5), K)
    a0 = rng.standard_normal((K, d)) * 0.5
    theta0 = np.concatenate([c0.reshape(-1), logR0, a0.reshape(-1)])
    theta, L, nst = adam(lambda th: loss_and_grad(th, u_tr, T_tr, K, d), theta0,
                         lr=LR, steps=steps, tol=TOL)
    return theta, L, nst


def fit_hypernet(u_tr, T_tr, d, H, rng, steps=NSTEPS):
    net = Hypernet(d, H, rng)

    def fg(p):
        net.set(p)
        return net.loss_grad(u_tr, T_tr)

    p, L, nst = adam(fg, net.params, lr=LR, steps=steps, tol=TOL)
    net.set(p)
    return net, L, nst


def _gf_unpack(th, K, d):
    return (th[:K * d].reshape(K, d), th[K * d:K * d + K],
            th[K * d + K:2 * K * d + K].reshape(K, d), th[2 * K * d + K:].reshape(K, d))


def gf_W(th, u, K, d):
    """ABLATION non intégrable : W(u) = I + Σ_k φ_k(u) a_k b_kᵀ (même base gaussienne, amplitudes libres)."""
    c, lr, a, b = _gf_unpack(th, K, d)
    R = np.exp(lr)
    D = u[:, None, :] - c[None]
    s_ = (D ** 2).sum(-1)
    phi = np.exp(-s_ / (2 * R ** 2))
    return np.eye(d)[None] + np.einsum("mk,ki,kj->mij", phi, a, b), (D, s_, phi, R)


def gf_loss_grad(th, u, T, K, d):
    c, lr, a, b = _gf_unpack(th, K, d)
    W, (D, s_, phi, R) = gf_W(th, u, K, d)
    E = W - T
    ga = np.einsum("mk,mij,kj->ki", phi, E, b)
    gb = np.einsum("mk,mij,ki->kj", phi, E, a)
    S = np.einsum("ki,mij,kj->mk", a, E, b)
    gc = np.einsum("mk,mkd->kd", S * phi / R ** 2, D)
    gr = np.sum(S * phi * s_ / R ** 2, axis=0)
    return 0.5 * float(np.sum(E ** 2)), np.concatenate([gc.ravel(), gr, ga.ravel(), gb.ravel()])


def fit_gaussfree(u_tr, T_tr, K, d, rng, steps=NSTEPS):
    th0 = np.concatenate([rng.standard_normal(K * d), rng.uniform(np.log(.3), np.log(1.5), K),
                          rng.standard_normal(2 * K * d) * .5])
    th, L, _ = adam(lambda t: gf_loss_grad(t, u_tr, T_tr, K, d), th0, lr=LR, steps=steps, tol=TOL)
    return th


def fit_hypernet_res(u_tr, T_tr, d, H, rng, steps=NSTEPS):
    """Hypernetwork RÉSIDUEL : W(u) = I + h(u) — même prior « proche de I » que le générateur."""
    net, L, nst = fit_hypernet(u_tr, T_tr - np.eye(d)[None], d, H, rng, steps)
    return (lambda u: net.W(u) + np.eye(d)[None]), L, nst


def lora_train_rmse(T_tr):
    """m LoRA indépendantes : par point, W = I + a wᵀ, (a, w) libres (2d params).
    Optimum en forme fermée : meilleure approximation de rang 1 de (T_i − I) par SVD."""
    d = T_tr.shape[1]
    sq = []
    for i in range(T_tr.shape[0]):
        s = np.linalg.svd(T_tr[i] - np.eye(d), compute_uv=False)
        sq.append(float(np.sum(s[1:] ** 2)))           # ‖E_i‖² = Σ_{r≥2} σ_r²
    return float(np.sqrt(np.mean(sq)))


# -------------------------------------------------------------------------------- métriques

def gen_rmse(Wfun, u_te, T_te):
    W = Wfun(u_te)
    return float(np.sqrt(np.mean((W - T_te) ** 2)))


def curl_rmse(Wfun, u_pts, d, h=1e-5):
    """Écart d'intégrabilité moyen : sqrt(mean_{j,k<l} (∂_l W_jk − ∂_k W_jl)²) par différences finies
    centrées, moyenné sur les points."""
    vals = []
    for u0 in u_pts[:NCURL]:
        D = np.zeros((d, d, d))                          # D[j,k,l] = ∂_l W_jk(u0)
        for l in range(d):
            ep = np.zeros(d)
            ep[l] = h
            Wp = Wfun((u0 + ep)[None])[0]
            Wm = Wfun((u0 - ep)[None])[0]
            D[:, :, l] = (Wp - Wm) / (2 * h)
        C = D - np.transpose(D, (0, 2, 1))              # C[j,k,l] = ∂_l W_jk − ∂_k W_jl
        ku, lu = np.triu_indices(d, 1)                   # paires k < l
        vals.append(float(np.sqrt(np.mean(C[:, ku, lu] ** 2))))
    return float(np.mean(vals))


def _stats(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r[k] is not None]
        out[f"{k}_med"] = float(np.median(v)) if v else None
        out[f"{k}_std"] = float(np.std(v)) if v else None
    return out


# -------------------------------------------------------------------------------- banc

def run_family(d, m, family, nseeds=NSEEDS):
    from scipy.stats import wilcoxon
    K, H = BUDGET[d]["K"], BUDGET[d]["H"]
    names = ["gen", "hyp", "hyp_res", "hyp_big", "gaussfree", "identity", "const_mean"]
    per = {n: [] for n in names}
    gen_rows, hyp_rows, lora_rows = [], [], []
    for s in range(nseeds):
        rng = np.random.default_rng([FAM_ID[family], d, m, s])
        u_tr, T_tr, u_te, T_te = make_target(family, d, m, rng)

        theta, L, nst = fit_generator(u_tr, T_tr, K, d, rng)
        Wg = lambda u: W_field(theta, u, K, d)
        gen_rows.append({
            "train": float(np.sqrt(2 * L / m)),
            "gen": gen_rmse(Wg, u_te, T_te) if T_te is not None else None,
            "curl": curl_rmse(Wg, u_te, d),
            "steps": float(nst),
        })

        net, Lh, nsth = fit_hypernet(u_tr, T_tr, d, H, rng)
        Wh = net.W
        hyp_rows.append({
            "train": float(np.sqrt(2 * Lh / m)),
            "gen": gen_rmse(Wh, u_te, T_te) if T_te is not None else None,
            "curl": curl_rmse(Wh, u_te, d),
            "steps": float(nsth),
        })
        lora_rows.append({"train": lora_train_rmse(T_tr)})

        if T_te is None:
            continue
        # --- contrôles (RNG dédiés : ne perturbent pas les tirages du générateur / hypernetwork) ---
        Wr, _, _ = fit_hypernet_res(u_tr, T_tr, d, H, np.random.default_rng([31, d, m, s]))
        Wb, _, _ = fit_hypernet_res(u_tr, T_tr, d, HBIG, np.random.default_rng([32, d, m, s]))
        tg = fit_gaussfree(u_tr, T_tr, KGF[d], d, np.random.default_rng([77, d, m, s]))
        I = np.eye(d)[None]
        per["gen"].append(gen_rows[-1]["gen"]); per["hyp"].append(hyp_rows[-1]["gen"])
        per["hyp_res"].append(gen_rmse(Wr, u_te, T_te)); per["hyp_big"].append(gen_rmse(Wb, u_te, T_te))
        per["gaussfree"].append(gen_rmse(lambda u: gf_W(tg, u, KGF[d], d)[0], u_te, T_te))
        per["identity"].append(float(np.sqrt(np.mean((T_te - I) ** 2))))
        per["const_mean"].append(float(np.sqrt(np.mean((T_te - T_tr.mean(0)) ** 2))))

    fam = {
        "generator": _stats(gen_rows, ["train", "gen", "curl", "steps"]),
        "hypernet": _stats(hyp_rows, ["train", "gen", "curl", "steps"]),
        "lora": _stats(lora_rows, ["train"]),
        "free": {"train_med": 0.0, "train_std": 0.0},   # fit exact par construction
    }
    if per["gen"]:
        fam["controls"] = {n: {"med": float(np.median(v)), "std": float(np.std(v)), "per_seed": v}
                           for n, v in per.items()}
        g = np.array(per["gen"])
        fam["tests"] = {}
        for n in ["hyp", "hyp_res", "hyp_big", "gaussfree", "identity", "const_mean"]:
            o = np.array(per[n])
            try:
                p = float(wilcoxon(g, o).pvalue)
            except ValueError:
                p = None
            fam["tests"][f"gen_vs_{n}"] = {"gen_wins": int((g < o).sum()), "n": len(g), "wilcoxon_p": p,
                                           "median_ratio_other_over_gen": float(np.median(o) / np.median(g))}
    return fam


def merge_save(d, m, fam_results):
    data = json.loads(OUT.read_text()) if OUT.exists() else {"configs": []}
    cfgs = data["configs"]
    K, H = BUDGET[d]["K"], BUDGET[d]["H"]
    cfg = next((c for c in cfgs if c["d"] == d and c["m"] == m), None)
    if cfg is None:
        cfg = {"d": d, "m": m, "K": K, "H": H,
               "budgets": {"generator": nparams(K, d),
                           "hypernet": H * d + H + d * d * H + d * d,
                           "lora": 2 * d * m, "free": d * d * m},
               "families": {}}
        cfgs.append(cfg)
    cfg["families"].update(fam_results)
    cfgs.sort(key=lambda c: (c["d"], c["m"]))
    data["configs"] = cfgs
    OUT.write_text(json.dumps(data, indent=1))
    print(f"[saved] d={d} m={m} familles={list(fam_results)} → {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--family", choices=["a", "b1", "b2", "mis"], default=None)
    args = ap.parse_args()
    fams = [args.family] if args.family else ["a", "mis", "b1", "b2"]
    out = {}
    for fam in fams:
        print(f"=== d={args.d} m={args.m} famille {fam} ===", flush=True)
        out[fam] = run_family(args.d, args.m, fam)
        print(json.dumps(out[fam], indent=1), flush=True)
    merge_save(args.d, args.m, out)


if __name__ == "__main__":
    main()
