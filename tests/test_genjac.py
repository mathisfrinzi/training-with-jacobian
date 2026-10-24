"""Auto-tests du prototype « matrices de poids = jacobiennes d'un générateur ».
Exécution : cd jacobian-weight-generator/tests && python3 test_genjac.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genjac import (W_field, loss_and_grad, phase1_loss_grad, lora_loss_grad, adam, nparams)
from hypernet import Hypernet


def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp)[0] - f(xm)[0]) / (2 * h)
    return g


def test_grads_multipoints():
    """Gradients analytiques du générateur = différences finies (d=2,4 ; K=3)."""
    rng = np.random.default_rng(0)
    for d in (2, 4):
        K, m = 3, 4
        theta = rng.standard_normal(nparams(K, d))
        u = rng.standard_normal((m, d))
        T = rng.standard_normal((m, d, d))
        _, g = loss_and_grad(theta, u, T, K, d)
        gn = _num_grad(lambda th: loss_and_grad(th, u, T, K, d), theta)
        assert np.max(np.abs(g - gn)) / (1 + np.max(np.abs(gn))) < 1e-5, (d, np.max(np.abs(g - gn)))


def test_grads_phase1():
    """Gradients analytiques de V1/V2/V3 = différences finies."""
    rng = np.random.default_rng(1)
    u0 = np.array([0.5, 0.5]); afix = np.array([0.6, 0.8])
    T = rng.standard_normal((2, 2))
    for variant, n in [("V1", 3), ("V2", 2), ("V3", 5)]:
        p = rng.standard_normal(n)
        _, g = phase1_loss_grad(p, u0, afix, T, variant)
        gn = _num_grad(lambda q: phase1_loss_grad(q, u0, afix, T, variant), p)
        assert np.max(np.abs(g - gn)) / (1 + np.max(np.abs(gn))) < 1e-5, variant


def test_grads_hypernet():
    """Gradients manuels de l'hypernetwork = différences finies."""
    rng = np.random.default_rng(2)
    d, H, m = 2, 3, 4
    net = Hypernet(d, H, rng)
    u = rng.standard_normal((m, d)); T = rng.standard_normal((m, d, d))
    _, g = net.loss_grad(u, T)
    gn = _num_grad(lambda th: (net.__class__.set(net, th) or net.loss_grad(u, T)), net.params.copy())
    net.set(net.params)
    assert np.max(np.abs(g - gn)) / (1 + np.max(np.abs(gn))) < 1e-5


def test_integrabilite():
    """Le champ W(u) = J_g(u) est intégrable : ∂_l W_jk = ∂_k W_jl aux différences finies,
    à la précision machine près (c'est la propriété que l'hypernetwork n'a pas)."""
    rng = np.random.default_rng(3)
    d, K = 3, 4
    theta = rng.standard_normal(nparams(K, d)) * 0.7
    u = rng.standard_normal((7, d))
    h = 1e-6
    curl = 0.0
    for i in range(u.shape[0]):
        for l in range(d):
            for k in range(d):
                if l == k:
                    continue
                el = np.zeros(d); el[l] = h
                ek = np.zeros(d); ek[k] = h
                # ∂_l W_jk(u) pour tous les j, comparé à ∂_k W_jl(u)
                Wp = W_field(theta, (u[i] + el)[None, :], K, d)[0]
                Wm = W_field(theta, (u[i] - el)[None, :], K, d)[0]
                dWl = (Wp - Wm) / (2 * h)              # (d,d) : [j, k]
                Wp = W_field(theta, (u[i] + ek)[None, :], K, d)[0]
                Wm = W_field(theta, (u[i] - ek)[None, :], K, d)[0]
                dWk = (Wp - Wm) / (2 * h)              # [j, l]
                curl = max(curl, float(np.max(np.abs(dWl[:, k] - dWk[:, l]))))
    assert curl < 1e-5, curl


def test_gap_e1b_theorique():
    """Gap théorique E1b : min_v ‖I + a vᵀ − W*‖_F = ‖(I − ââᵀ)(W* − I)‖_F. Vérifié par LoRA
    à a fixé (w libre), dont l'optimum est w* = (W*−I)ᵀa/‖a‖²."""
    rng = np.random.default_rng(4)
    a = np.array([0.6, 0.8])
    for _ in range(10):
        T = rng.standard_normal((2, 2))
        G = T - np.eye(2)
        P = np.eye(2) - np.outer(a, a)
        gap_th = np.sqrt(np.sum((P @ G) ** 2))
        w0 = rng.standard_normal(2)
        w, L, _ = adam(lambda w: lora_loss_grad(w, a, T), w0, lr=0.05, steps=8000, tol=1e-16)
        gap_mes = np.sqrt(2 * L)
        assert abs(gap_mes - gap_th) / gap_th < 1e-6, (gap_mes, gap_th)


def test_ablation_gaussfree_et_controles_phase2():
    """Gradient analytique de l'ablation non intégrable (W = I + Σ φₖ aₖbₖᵀ) = différences finies ;
    cible « mis » (tanh) réellement intégrable (curl ~ 0) ; baselines triviales bien définies."""
    import phase2 as P2
    rng = np.random.default_rng(0)
    K, d = 3, 3
    th = rng.standard_normal(K * (3 * d + 1)) * 0.5
    u = rng.standard_normal((5, d)); T = rng.standard_normal((5, d, d))
    _, g = P2.gf_loss_grad(th, u, T, K, d)
    fd = _num_grad(lambda t: P2.gf_loss_grad(t, u, T, K, d), th)
    assert np.max(np.abs(fd - g)) < 1e-6, np.max(np.abs(fd - g))
    # cible « mis » : champ de jacobiennes de g*(u) = u + ½ B tanh(Cu + e) ⇒ curl nul
    r = np.random.default_rng(1)
    u_tr, T_tr, u_te, T_te = P2.make_target("mis", 4, 4, r)
    # reconstruire J à partir de la même graine pour tester le curl par différences finies
    r = np.random.default_rng(1)
    d_, h = 4, 3
    r.uniform(-P2.UBOX, P2.UBOX, size=(4, d_)); r.uniform(-P2.UBOX, P2.UBOX, size=(P2.NTEST, d_))
    B_ = r.standard_normal((d_, h)) / np.sqrt(h); C_ = r.standard_normal((h, d_)) / np.sqrt(d_) * 1.5
    e_ = r.standard_normal(h) * 0.5

    def J(uu):
        tt = np.tanh(uu @ C_.T + e_)
        return np.eye(d_)[None] + 0.5 * np.einsum("ih,mh,hj->mij", B_, 1 - tt ** 2, C_)

    assert P2.curl_rmse(J, u_te, d_) < 1e-7
    assert np.allclose(J(u_te), T_te)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} tests passés.")
