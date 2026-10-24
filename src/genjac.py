"""Générateurs jacobiens : W(u) = J_g(u) avec g(u) = u + Σ_k a_k·φ_k(u), φ gaussienne.

φ_k(u) = exp(−‖u − c_k‖² / 2R_k²),  W(u) = I + Σ_k a_k·v_k(u)ᵀ,  v_k(u) = −(φ_k(u)/R_k²)(u − c_k).

Paramètres (budget K(2d+1) conformément au plan) : centres c_k (K,d), log R_k (K,), vecteurs a_k (K,d).
Toutes les dérivées sont analytiques et vérifiées contre les différences finies dans les tests.

Dérivations (une gaussienne, un point u, dvec = u − c, s = ‖dvec‖², φ = exp(−s/2R²), q = φ/R²) :
    v = −q·dvec
    ∂v/∂c_j   = q·e_j − (φ·dvec_j/R⁴)·dvec         (∂φ/∂c_j = φ·dvec_j/R²  car ∂dvec/∂c_j = −e_j)
    ∂v/∂log R = −φ·dvec·(s − 2R²)/R⁴               (via ∂φ/∂R = φ·s/R³)
Perte L = Σ_i ½‖W(u_i) − T_i‖²_F,  E_i = W(u_i) − T_i,  p_i = E_iᵀ·a_k :
    ∂L/∂a_k   = Σ_i E_i·v_k(u_i)
    ∂L/∂c_kj  = Σ_i p_i ⊙ ∂v_k/∂c_j(u_i) ;  ∂L/∂logR_k = Σ_i p_i ⊙ ∂v_k/∂logR(u_i)
"""
import numpy as np


def unpack(theta, K, d):
    i = K * d
    c = theta[:i].reshape(K, d)
    logR = theta[i:i + K]
    a = theta[i + K:].reshape(K, d)
    return c, logR, a


def nparams(K, d):
    return K * (2 * d + 1)


def v_k(u, c, logR):
    """v_k(u) = −(φ/R²)(u − c) pour un lot u (m,d). Retourne (m,d)."""
    R = np.exp(logR)
    dvec = u - c
    phi = np.exp(-np.sum(dvec ** 2, axis=1) / (2 * R ** 2))
    return -(phi / R ** 2)[:, None] * dvec


def W_field(theta, u, K, d):
    """W(u_i) = I + Σ_k a_k v_k(u_i)ᵀ. Retourne (m,d,d)."""
    c, logR, a = unpack(theta, K, d)
    m = u.shape[0]
    W = np.tile(np.eye(d), (m, 1, 1))
    for k in range(K):
        vk = v_k(u, c[k], logR[k])
        W += a[k][None, :, None] * vk[:, None, :]
    return W


def loss_and_grad(theta, u, T, K, d):
    """L = Σ_i ½‖W(u_i) − T_i‖²_F et gradient analytique (vectorisé sur les points). Retourne (L, grad)."""
    c, logR, a = unpack(theta, K, d)
    m = u.shape[0]
    W = np.tile(np.eye(d), (m, 1, 1))
    vs = np.stack([v_k(u, c[k], logR[k]) for k in range(K)])      # (K,m,d)
    for k in range(K):
        W += a[k][None, :, None] * vs[k][:, None, :]
    E = W - T
    L = 0.5 * float(np.sum(E ** 2))
    gc = np.zeros((K, d)); gr = np.zeros(K); ga = np.zeros((K, d))
    for k in range(K):
        ga[k] = np.einsum("mij,mj->i", E, vs[k])
        R = float(np.exp(logR[k]))
        dvec = u - c[k]                                           # (m,d)
        s = np.sum(dvec ** 2, axis=1)                             # (m,)
        phi = np.exp(-s / (2 * R ** 2))                           # (m,)
        q = phi / R ** 2
        p = np.einsum("mij,i->mj", E, a[k])                       # (m,d) : p_i = E_iᵀ a_k
        sc = np.einsum("md,md->m", dvec, p)                       # (m,) : dvec_i · p_i
        gc[k] = (q[:, None] * p).sum(axis=0) - ((phi / R ** 4) * sc) @ dvec
        gr[k] = -float(np.sum((phi * (s - 2 * R ** 2) / R ** 4) * sc))
    grad = np.concatenate([gc.reshape(-1), gr, ga.reshape(-1)])
    return L, grad


# ====================================================================================== Phase 1

def phase1_loss_grad(params, u0, afix, T, variant, R0=0.5):
    """Phase 1 : UNE gaussienne, UN point u0, perte ½‖W − T‖²_F. W = I + a vᵀ, v = −(φ/R²)(u0 − c).
    variantes : V1 = (x, y, logR), a fixé ; V2 = (x, y), R = R₀ fixé, a fixé ;
                V3 = (x, y, logR, a(2)) avec a libre. Retourne (L, grad)."""
    if variant == "V2":
        x, y = params; logR = np.log(R0); a = afix
    elif variant == "V1":
        x, y, logR = params; a = afix
    elif variant == "V3":
        x, y, logR = params[0], params[1], params[2]; a = params[3:5]
    else:
        raise ValueError(variant)
    d = 2
    c = np.array([x, y])
    R = np.exp(logR)
    dvec = u0 - c
    s = float(dvec @ dvec)
    phi = np.exp(-s / (2 * R ** 2))
    q = phi / R ** 2
    v = -q * dvec
    W = np.eye(d) + np.outer(a, v)
    E = W - T
    L = 0.5 * float(np.sum(E ** 2))
    p = E.T @ a
    dvc = q * np.eye(d) - (phi / R ** 4) * np.outer(dvec, dvec)   # colonne j : ∂v/∂c_j
    gc = dvc @ p
    dvr = -phi * dvec * (s - 2 * R ** 2) / R ** 4
    parts = [gc[0], gc[1]]
    if variant in ("V1", "V3"):
        parts.append(float(p @ dvr))
    if variant == "V3":
        parts.extend(list(E @ v))                                  # ∂L/∂a = E v
    return L, np.array(parts)


def lora_loss_grad(w, afix, T):
    """Baseline LoRA rang 1 à a fixé : W = I + a wᵀ, w ∈ ℝ² libre. Retourne (L, ∇w)."""
    W = np.eye(2) + np.outer(afix, w)
    E = W - T
    return 0.5 * float(np.sum(E ** 2)), E.T @ afix


def adam(grad_fn, x0, lr=0.05, steps=6000, tol=1e-13, b1=0.9, b2=0.999, eps=1e-8):
    """Adam minimal. grad_fn : x → (L, grad). Retourne (x, L, n_steps_effectifs)."""
    x = np.asarray(x0, float).copy()
    mm = np.zeros_like(x); vv = np.zeros_like(x)
    L = None
    for t in range(1, steps + 1):
        L, g = grad_fn(x)
        if L < tol:
            return x, L, t - 1
        mm = b1 * mm + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        mh = mm / (1 - b1 ** t); vh = vv / (1 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
    L, _ = grad_fn(x)
    return x, L, steps
