"""Tests Phase 3 — gradients complets (transformer + fournisseurs) contre différences finies.

Config minuscule en float64 : V=9, d=6, d_ff=10, L=2, H=2, T=8, B=2. La perte totale (entropie
croisée d'un lot fixe) est perturbée paramètre par paramètre (différences centrées, h = 1e-6) et
comparée aux gradients analytiques : ceux du transformer (backward manuel) et ceux du fournisseur
(provider.backprop(Gs), chaîne exacte ∂L/∂θ = Σℓ ∂L/∂W1ℓ · ∂W1ℓ/∂θ). Tous les paramètres des
fournisseurs sont rendus non nuls avant le test (sinon LoRA teste des gradients morts à 0).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from phase3_model import Transformer
from phase3_methods import make_provider
from phase3_corpus import get_task, make_corpus_a, make_corpus_b

V, D, D_FF, L, H, T, B = 9, 6, 10, 2, 2, 8, 2
DT = np.float64


def tiny_setup(seed=0, method="gen"):
    rng = np.random.default_rng(seed)
    model = Transformer(V, D, D_FF, L, H, T, rng, dtype=DT)
    prov = make_provider(method, D, D_FF, L, rng, dtype=DT, K=3, r=2, e_dim=4, Hh=5)
    # rendre tous les paramètres du fournisseur non nuls et actifs
    for k, v in prov.opt.params.items():
        r2 = np.random.default_rng(seed + 1)
        prov.opt.params[k] = v + (r2.standard_normal(v.shape) * 0.1).astype(DT)
    X = np.array([[rng.integers(0, V) for _ in range(T)] for _ in range(B)])
    return model, prov, X


def total_loss(model, prov, X):
    return model.forward(X, prov.W_layers())[0]


def fd_check(tag, model, prov, X, entries, h=1e-6, tol=2e-4):
    loss0 = total_loss(model, prov, X)
    W1s = prov.W_layers()
    grads, Gs = model.backward(model.forward(X, W1s)[1], W1s)
    prov.backprop(Gs)
    prov_grads = {k: v.copy() for k, v in prov._g.items()}
    ok, worst = True, 0.0
    for arr, idx, name, analytic in entries:
        orig = arr[idx]
        arr[idx] = orig + h
        lp = total_loss(model, prov, X)
        arr[idx] = orig - h
        lm = total_loss(model, prov, X)
        arr[idx] = orig
        fd = (lp - lm) / (2 * h)
        err = abs(fd - analytic[idx]) / max(1e-8, abs(fd) + abs(analytic[idx]))
        worst = max(worst, err)
        if err > tol:
            ok = False
            print(f"  ÉCHEC {tag}/{name}{idx}: fd={fd:.6e} analytique={analytic[idx]:.6e}")
    print(f"  {tag}: {'OK' if ok else 'ÉCHEC'} (pire erreur relative {worst:.2e})")
    return ok


def test_backward_transformer():
    model, prov, X = tiny_setup(0, "dense")
    W1s = prov.W_layers()
    _, cache = model.forward(X, W1s)
    grads, _ = model.backward(cache, W1s)
    entries = [(model.params["E"], (2, 3), "E", grads["E"]),
               (model.params["P"], (4, 1), "P", grads["P"]),
               (model.params["gf"], (1,), "gf", grads["gf"]),
               (model.params["Wq_0"], (2, 3), "Wq0", grads["Wq_0"]),
               (model.params["Wk_1"], (1, 4), "Wk1", grads["Wk_1"]),
               (model.params["Wv_0"], (0, 0), "Wv0", grads["Wv_0"]),
               (model.params["Wp_0"], (3, 2), "Wp0", grads["Wp_0"]),
               (model.params["g1_0"], (4,), "g1_0", grads["g1_0"]),
               (model.params["b2_1"], (2,), "b2_1(LN2)", grads["b2_1"]),
               (model.params["b1f_0"], (5,), "b1f0", grads["b1f_0"]),
               (model.params["W2_1"], (1, 2), "W2_1", grads["W2_1"]),
               (model.params["b2f_0"], (3,), "b2f0", grads["b2f_0"])]
    assert fd_check("transformer", model, prov, X, entries)


def test_provider_gen():
    for method in ["gen", "gen_naive", "genfree", "lora", "hyp", "albert", "dense"]:
        model, prov, X = tiny_setup(1, method)
        entries = []
        r2 = np.random.default_rng(5)
        for k, v in prov.opt.params.items():
            for _ in range(2):
                idx = tuple(r2.integers(0, s) for s in v.shape)
                entries.append((prov.opt.params[k], idx, k, prov._g[k]))
        assert fd_check(f"fournisseur {method}", model, prov, X, entries)


def test_corpora():
    ta = make_corpus_a(5000, seed=0)
    tb = make_corpus_b(5000, seed=0)
    ta2 = make_corpus_a(5000, seed=0)
    assert ta == ta2, "corpus A doit être déterministe"
    assert len(ta) >= 4000 and len(tb) >= 4000
    assert set(tb) <= set("fnr x()_=+*-\n0123456789"), f"alphabet B inattendu: {sorted(set(tb))}"
    assert len(set(ta)) >= 20   # 13 cons. + 5 voyelles max + " ." ","
    tr, va = get_task("a", 20_000)
    assert tr.V == va.V and tr.V >= 20 and len(va.data) < len(tr.data)
    X = tr.sample_batch(4, 64, np.random.default_rng(0))
    assert X.shape == (4, 64) and X.max() < tr.V
    print(f"  corpus A: V={tr.V}, corpus B: V={get_task('b', 1000)[0].V}")
    print("  corpora: OK")


def test_budgets():
    rng = np.random.default_rng(0)
    d, d_ff, L = 32, 64, 4
    bs = {}
    for m in ["gen", "gen_naive", "genfree", "lora", "hyp", "albert", "dense"]:
        p = make_provider(m, d, d_ff, L, rng, K=6, r=2, e_dim=8, Hh=6)
        bs[m] = p.n_params
    print(f"  budgets W1 : {bs}")
    assert abs(bs["gen"] - 2758) == 0 and abs(bs["lora"] - 2816) == 0
    assert bs["genfree"] == 2821 and max(bs["genfree"], 2816) / min(bs["genfree"], 2758) < 1.03
    assert bs["hyp"] == 2806 and bs["albert"] == 2048 and bs["dense"] == 8192
    assert max(bs["gen"], bs["lora"], bs["hyp"]) / min(bs["gen"], bs["lora"], bs["hyp"]) < 1.03
    print("  budgets: OK (appariés à ±4 %, dense 2,9× plus, ALBERT sous-budget — hypernet H=6 pour égaliser)")


def test_live_init_and_integrability():
    """Garde-fou contre le défaut de l'init naïve : à l'init, les gaussiennes doivent être VIVANTES
    (φ_max ≥ 0,1 et ΔW non négligeable) en dimension 32 ET 128, et le champ du générateur doit rester
    intégrable (curl ~ 0) tandis que celui de l'ablation ne l'est pas."""
    for d in (32, 128):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            prov = make_provider("gen", d, 2 * d, 4, rng, dtype=np.float64, K=6)
            p = prov.opt.params
            R = np.exp(p["logR"])
            phi = np.array([[np.exp(-np.sum((p["U"][l] - p["C"][k]) ** 2) / (2 * R[k] ** 2))
                             for k in range(6)] for l in range(4)])
            assert phi.max() > 0.1, f"gaussiennes mortes à l'init (d={d}, graine {seed}) : {phi.max():.1e}"
            assert (phi > 0.01).sum() >= 4
        leg = make_provider("gen_naive", d, 2 * d, 4, np.random.default_rng(0), dtype=np.float64, K=6)
        q = leg.opt.params
        Rl = np.exp(q["logR"])
        phil = max(np.exp(-np.sum((q["U"][l] - q["C"][k]) ** 2) / (2 * Rl[k] ** 2))
                   for l in range(4) for k in range(6))
        print(f"  d={d}: φ_max live > 0,1 ; naive {phil:.1e}")
    # intégrabilité : ∂ₗ W_jk − ∂ₖ W_jl pour W(u) = J_g(u) (formule v_k) vs ablation φ(u)·a bᵀ
    from genjac import W_field
    rng = np.random.default_rng(3)
    d, K = 4, 3
    th = np.concatenate([rng.standard_normal(K * d), rng.uniform(-0.5, 0.5, K), rng.standard_normal(K * d) * 0.5])
    free_b = rng.standard_normal((K, d)) * 0.5
    a = th[K * d + K:].reshape(K, d); c = th[:K * d].reshape(K, d); R = np.exp(th[K * d:K * d + K])

    def Wfree(u):
        phi = np.exp(-((u - c) ** 2).sum(1) / (2 * R ** 2))
        return np.eye(d) + np.einsum("k,ki,kj->ij", phi, a, free_b)

    def curl(Wf, u, h=1e-5):
        D = np.zeros((d, d, d))
        for l in range(d):
            e = np.zeros(d); e[l] = h
            D[:, :, l] = (Wf(u + e) - Wf(u - e)) / (2 * h)
        return np.abs(D - np.transpose(D, (0, 2, 1))).max()

    u0 = rng.standard_normal(d) * 0.5
    c_gen = curl(lambda u: W_field(th, u[None], K, d)[0], u0)
    c_free = curl(Wfree, u0)
    print(f"  curl générateur {c_gen:.1e} ; curl ablation {c_free:.1e}")
    assert c_gen < 1e-8 and c_free > 1e-3
    print("  init vivante + intégrabilité: OK")


if __name__ == "__main__":
    test_backward_transformer()
    test_provider_gen()
    test_corpora()
    test_budgets()
    test_live_init_and_integrability()
    print("5 tests passés.")
