"""Phase 3 — Les cinq fournisseurs de la première matrice du FFN, W1 (d_ff, d) par couche.

Budget total (paramètres consacrés aux L matrices W1, tous incluent la matrice partagée M) :
  - générateur jacobien : M + K gaussiennes (aₖ d_ff, cₖ d, logRₖ) + points de couche uℓ (L,d)
        Wℓ = M + Σₖ aₖ vₖ(uℓ)ᵀ,  vₖ(u) = −(φₖ(u)/Rₖ²)(u − cₖ)   — la famille est INTÉGRABLE ;
  - L LoRA indépendantes (Hu et al. 2021) : M + (Aℓ, Bℓ) libres par couche, rang r, sans liaison ;
  - hypernetwork à plongement de couche : MLP(eℓ) → (âℓ, ŵℓ), Wℓ = M + âℓŵℓᵀ — sortie libre
        de rang 1 générée par couche (aucune contrainte d'intégrabilité) ;
  - ALBERT (Lan et al. 2020) : Wℓ = W partagé à l'identique ;
  - dense : L matrices libres (plafond de référence, budget ≫).

Les gradients du générateur sont analytiques (formules de la Phase 1, rôles u/c échangés au signe
près : ∂v/∂u = −∂v/∂c) ; tous vérifiés contre les différences finies dans tests/test_phase3.py.
"""
import numpy as np


class Adamizer:
    """Adam minimal sur un dictionnaire de paramètres."""

    def __init__(self, params):
        self.params = params
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            self.params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)


def _v(ku, c, R):
    """v(u) = −(φ/R²)(u − c). Retourne (v, dvec, s, phi, R)."""
    dvec = ku - c
    s = float(dvec @ dvec)
    phi = np.exp(-s / (2 * R * R))
    return -(phi / R ** 2) * dvec, dvec, s, phi, R


def _live_geometry(d, d_ff, L, K, rng):
    """Init « vivante », indépendante de la dimension : toutes les distances sont fixées en NORME
    (et non coordonnée par coordonnée), si bien que φ(uℓ − cₖ) ~ O(1) quelle que soit d.
    Points de couche : alignés le long d'une direction unitaire, espacés de 0,5 (+ bruit de norme 0,05) ;
    centres : posés près des points de couche (bruit de norme 0,3) ; rayons ~ U(0,5 ; 1,5)."""
    e = rng.standard_normal(d)
    e /= np.linalg.norm(e)
    U = np.stack([(l - (L - 1) / 2) * 0.5 * e + 0.05 / np.sqrt(d) * rng.standard_normal(d)
                  for l in range(L)])
    C = np.stack([U[k % L] + 0.3 / np.sqrt(d) * rng.standard_normal(d) for k in range(K)])
    return {
        "M": rng.standard_normal((d_ff, d)) * np.sqrt(2.0 / (d + d_ff)),
        "C": C,
        "logR": rng.uniform(np.log(0.5), np.log(1.5), K),
        "U": U,
    }


class Generator:
    """Wℓ = M + Σₖ aₖ vₖ(uℓ)ᵀ — gaussiennes, gradients analytiques."""

    name = "gen"

    def __init__(self, d, d_ff, L, K, rng, dtype=np.float32, init="live"):
        self.d, self.d_ff, self.L, self.K = d, d_ff, L, K
        if init == "naive":
            # Init naïve (ablation) : centres ~ 0,8·N(0,1) en dimension d.
            # ‖u − c‖² ~ d·0,64 ⇒ φ ~ 1e-6…1e-18 en d = 32 : gaussiennes quasi mortes au départ.
            p = {
                "M": rng.standard_normal((d_ff, d)) * np.sqrt(2.0 / (d + d_ff)),
                "A": rng.standard_normal((K, d_ff)) * 0.02,
                "C": rng.standard_normal((K, d)) * 0.8,
                "logR": rng.uniform(np.log(0.3), np.log(1.5), K),
                "U": np.stack([(l - (L - 1) / 2) * 0.5 * np.ones(d)
                               + rng.standard_normal(d) * 0.05 for l in range(L)]),
            }
        else:
            p = _live_geometry(d, d_ff, L, K, rng)
            p["A"] = rng.standard_normal((K, d_ff)) * 0.02
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}

    @property
    def n_params(self):
        p = self.opt.params
        return int(p["M"].size + p["A"].size + p["C"].size + p["logR"].size + p["U"].size)

    def W_layers(self):
        p = self.opt.params
        M, A, C, U = p["M"], p["A"], p["C"], p["U"]
        R = np.exp(p["logR"])
        Ws = []
        for l in range(self.L):
            W = M.copy()
            for k in range(self.K):
                v, *_ = _v(U[l], C[k], float(R[k]))
                W += np.outer(A[k], v)
            Ws.append(W)
        return Ws

    def backprop(self, Gs):
        p = self.opt.params
        M, A, C, U = p["M"], p["A"], p["C"], p["U"]
        R = np.exp(p["logR"])
        g = self._g
        for arr in g.values():
            arr.fill(0)
        g["M"] += sum(Gs)
        for l in range(self.L):
            for k in range(self.K):
                Rk = float(R[k])
                v, dvec, s, phi, _ = _v(U[l], C[k], Rk)
                g["A"][k] += Gs[l] @ v                       # ∂L/∂aₖ
                dv = Gs[l].T @ A[k]                          # ∂L/∂vₖ(uℓ) (d,)
                q = phi / Rk ** 2
                sc = float(dvec @ dv)
                J_dv_du = -q * dv + (phi / Rk ** 4) * sc * dvec   # ∂v/∂u = −∂v/∂c
                J_dv_dc = q * dv - (phi / Rk ** 4) * sc * dvec
                g["U"][l] += J_dv_du
                g["C"][k] += J_dv_dc
                g["logR"][k] += float(dv @ (-phi * dvec * (s - 2 * Rk ** 2) / Rk ** 4))

    def step(self, lr):
        self.opt.step(self._g, lr)


class GenFree:
    """ABLATION non intégrable : Wℓ = M + Σₖ φₖ(uℓ)·aₖ bₖᵀ — MÊME base gaussienne φₖ(u) = exp(−‖u−cₖ‖²/2Rₖ²),
    même géométrie de départ, mêmes points de couche uℓ, mais l'amplitude de chaque gaussienne est une
    matrice de rang 1 LIBRE aₖbₖᵀ au lieu de aₖ·(−(φ/R²)(u−c))ᵀ. Le champ u ↦ W(u) n'est plus un champ de
    jacobiennes : il isole l'apport de la contrainte d'intégrabilité (K−1 gaussiennes pour apparier le budget)."""

    name = "genfree"

    def __init__(self, d, d_ff, L, K, rng, dtype=np.float32):
        self.d, self.d_ff, self.L, self.K = d, d_ff, L, K
        p = _live_geometry(d, d_ff, L, K, rng)
        p["A"] = rng.standard_normal((K, d_ff)) * 0.02
        p["B"] = rng.standard_normal((K, d)) * (0.3 / np.sqrt(d))
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}

    @property
    def n_params(self):
        return int(sum(v.size for v in self.opt.params.values()))

    def _phi(self, u, c, R):
        dvec = u - c
        s = float(dvec @ dvec)
        return np.exp(-s / (2 * R * R)), dvec, s

    def W_layers(self):
        p = self.opt.params
        R = np.exp(p["logR"])
        Ws = []
        for l in range(self.L):
            W = p["M"].copy()
            for k in range(self.K):
                phi, _, _ = self._phi(p["U"][l], p["C"][k], float(R[k]))
                W += phi * np.outer(p["A"][k], p["B"][k])
            Ws.append(W)
        return Ws

    def backprop(self, Gs):
        p = self.opt.params
        R = np.exp(p["logR"])
        g = self._g
        for arr in g.values():
            arr.fill(0)
        g["M"] += sum(Gs)
        for l in range(self.L):
            for k in range(self.K):
                Rk = float(R[k])
                phi, dvec, s = self._phi(p["U"][l], p["C"][k], Rk)
                G = Gs[l]
                g["A"][k] += phi * (G @ p["B"][k])
                g["B"][k] += phi * (G.T @ p["A"][k])
                dphi = float(p["A"][k] @ G @ p["B"][k])
                g["U"][l] += dphi * (-phi * dvec / Rk ** 2)
                g["C"][k] += dphi * (phi * dvec / Rk ** 2)
                g["logR"][k] += dphi * phi * s / Rk ** 2

    def step(self, lr):
        self.opt.step(self._g, lr)


class Lora:
    """Wℓ = M + AℓBℓᵀ, rang r par couche, facteurs libres (B init à 0 → Wℓ = M au départ)."""

    name = "lora"

    def __init__(self, d, d_ff, L, r, rng, dtype=np.float32):
        self.d, self.d_ff, self.L, self.r = d, d_ff, L, r
        p = {
            "M": rng.standard_normal((d_ff, d)) * np.sqrt(2.0 / (d + d_ff)),
            "A": rng.standard_normal((L, d_ff, r)) * 0.1,
            "B": np.zeros((L, d, r)),
        }
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}

    @property
    def n_params(self):
        p = self.opt.params
        return int(p["M"].size + p["A"].size + p["B"].size)

    def W_layers(self):
        p = self.opt.params
        return [p["M"] + p["A"][l] @ p["B"][l].T for l in range(self.L)]

    def backprop(self, Gs):
        p = self.opt.params
        g = self._g
        g["M"].fill(0)
        g["M"] += sum(Gs)
        for l in range(self.L):
            g["A"][l].fill(0)
            g["B"][l].fill(0)
            g["A"][l] += Gs[l] @ p["B"][l]
            g["B"][l] += Gs[l].T @ p["A"][l]

    def step(self, lr):
        self.opt.step(self._g, lr)


class Hypernet:
    """MLP(eℓ) → tanh → (âℓ, ŵℓ) ; Wℓ = M + âℓŵℓᵀ. Plongement de couche appris."""

    name = "hyp"

    def __init__(self, d, d_ff, L, e_dim, H, rng, dtype=np.float32):
        self.d, self.d_ff, self.L = d, d_ff, L
        p = {
            "M": rng.standard_normal((d_ff, d)) * np.sqrt(2.0 / (d + d_ff)),
            "Emb": rng.standard_normal((L, e_dim)) * 0.5,
            "W1": rng.standard_normal((e_dim, H)) * np.sqrt(1.0 / e_dim),
            "b1": np.zeros(H),
            "Wa": rng.standard_normal((H, d_ff)) * 0.02,
            "ba": np.zeros(d_ff),
            "Ww": rng.standard_normal((H, d)) * 0.02,
            "bw": np.zeros(d),
        }
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}
        self._h = None

    @property
    def n_params(self):
        p = self.opt.params
        return int(sum(p[k].size for k in p if k != "M") + p["M"].size)

    def W_layers(self):
        p = self.opt.params
        hs = []
        Ws = []
        for l in range(self.L):
            h = np.tanh(p["Emb"][l] @ p["W1"] + p["b1"])
            hs.append(h)
            a = h @ p["Wa"] + p["ba"]
            w = h @ p["Ww"] + p["bw"]
            Ws.append(p["M"] + np.outer(a, w))
        self._h = hs
        return Ws

    def backprop(self, Gs):
        p = self.opt.params
        g = self._g
        for arr in g.values():
            arr.fill(0)
        g["M"] += sum(Gs)
        for l in range(self.L):
            h = self._h[l]
            a = h @ p["Wa"] + p["ba"]
            w = h @ p["Ww"] + p["bw"]
            da = Gs[l] @ w                                  # (d_ff,)
            dw = Gs[l].T @ a                                # (d,)
            g["Wa"] += np.outer(h, da)
            g["ba"] += da
            g["Ww"] += np.outer(h, dw)
            g["bw"] += dw
            dh = p["Wa"] @ da + p["Ww"] @ dw
            dh_pre = dh * (1 - h * h)
            g["W1"] += np.outer(p["Emb"][l], dh_pre)
            g["b1"] += dh_pre
            g["Emb"][l] += p["W1"] @ dh_pre

    def step(self, lr):
        self.opt.step(self._g, lr)


class Albert:
    """Wℓ = W partagé à l'identique entre les L couches (partage total, aucune adaptation)."""

    name = "albert"

    def __init__(self, d, d_ff, L, rng, dtype=np.float32):
        self.L = L
        p = {"W": rng.standard_normal((d_ff, d)) * np.sqrt(2.0 / (d + d_ff))}
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}

    @property
    def n_params(self):
        return int(self.opt.params["W"].size)

    def W_layers(self):
        W = self.opt.params["W"]
        return [W] * self.L

    def backprop(self, Gs):
        self._g["W"].fill(0)
        self._g["W"] += sum(Gs)

    def step(self, lr):
        self.opt.step(self._g, lr)


class Dense:
    """L matrices libres — plafond de référence (budget ≫ méthodes à budget apparié)."""

    name = "dense"

    def __init__(self, d, d_ff, L, rng, dtype=np.float32):
        self.L = L
        p = {"W": (rng.standard_normal((L, d_ff, d)) * np.sqrt(2.0 / (d + d_ff)))}
        self.opt = Adamizer({k: v.astype(dtype) for k, v in p.items()})
        self._g = {k: np.zeros_like(v) for k, v in self.opt.params.items()}

    @property
    def n_params(self):
        return int(self.opt.params["W"].size)

    def W_layers(self):
        return list(self.opt.params["W"])

    def backprop(self, Gs):
        self._g["W"].fill(0)
        for l in range(self.L):
            self._g["W"][l] += Gs[l]

    def step(self, lr):
        self.opt.step(self._g, lr)


def make_provider(method, d, d_ff, L, rng, dtype=np.float32, K=6, r=2, e_dim=8, Hh=24):
    if method == "gen":                      # init vivante (géométrie fixée en norme)
        return Generator(d, d_ff, L, K, rng, dtype, init="live")
    if method == "gen_naive":               # init naïve (ablation : gaussiennes mortes)
        return Generator(d, d_ff, L, K, rng, dtype, init="naive")
    if method == "genfree":                  # ablation non intégrable, budget apparié (K−1 gaussiennes)
        return GenFree(d, d_ff, L, K - 1, rng, dtype)
    if method == "lora":
        return Lora(d, d_ff, L, r, rng, dtype)
    if method == "hyp":
        return Hypernet(d, d_ff, L, e_dim, Hh, rng, dtype)
    if method == "albert":
        return Albert(d, d_ff, L, rng, dtype)
    if method == "dense":
        return Dense(d, d_ff, L, rng, dtype)
    raise ValueError(method)
