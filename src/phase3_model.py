"""Phase 3 — Petit transformer GPT-like en NumPy pur, rétropropagation manuelle complète.

Architecture : embeddings jetons + positions, L blocs pre-LN (attention causale à H têtes, FFN
ReLU), LayerNorm final, tête liée aux embeddings (weight tying). Perte : entropie croisée du jeton
suivant, moyenne sur les positions valides.

La première matrice du FFN de chaque couche, W1 (d_ff, d), est FOURNIE par un « fournisseur »
(générateur jacobien, LoRA, hypernetwork, ALBERT ou dense) : le backward du transformer calcule
∂L/∂W1ℓ pour chaque couche et le transmet au fournisseur, qui rétropropage dans ses propres
paramètres. Tous les autres paramètres (attention, W2, LN, embeddings) sont gérés ici.

Structure d'un bloc (pre-LN), x_in l'entrée du bloc :
    h1 = LN1(x_in) ; x_mid = x_in + Attn(h1) ; h2 = LN2(x_mid) ;
    z = relu(h2·W1ᵀ + b1f) ; x_out = x_mid + (z·W2ᵀ + b2f)
Gradients vérifiés contre les différences finies dans tests/test_phase3.py.
"""
import numpy as np


def ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * rstd
    return xh * g + b, xh, rstd


def ln_backward(dout, xh, rstd, g):
    """dout a la forme de xh. Retourne (dx, dg, db) — dg, db sommés sur (batch, temps)."""
    dg = (dout * xh).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    dxh = dout * g
    m1 = dxh.mean(-1, keepdims=True)
    m2 = (dxh * xh).mean(-1, keepdims=True)
    dx = (dxh - m1 - xh * m2) * rstd
    return dx, dg, db


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


class Transformer:
    """V jetons, T positions, L couches, d modèle, d_ff FFN, H têtes."""

    def __init__(self, V, d, d_ff, L, H, T, rng, dtype=np.float32):
        self.V, self.d, self.d_ff, self.L, self.H, self.T = V, d, d_ff, L, H, T
        self.dh = d // H
        self.dtype = dtype
        s_attn = 1.0 / np.sqrt(d)
        s_ffn = np.sqrt(2.0 / (d + d_ff))
        p = {
            "E": rng.standard_normal((V, d)) * 0.1,
            "P": rng.standard_normal((T, d)) * 0.02,
            "gf": np.ones(d), "bf": np.zeros(d),
        }
        for l in range(L):
            p[f"g1_{l}"] = np.ones(d);   p[f"b1_{l}"] = np.zeros(d)     # LN1
            p[f"g2_{l}"] = np.ones(d);   p[f"b2_{l}"] = np.zeros(d)     # LN2
            for w in ("q", "k", "v", "p"):
                p[f"W{w}_{l}"] = rng.standard_normal((d, d)) * s_attn
            p[f"b1f_{l}"] = np.zeros(d_ff)                              # biais entrée FFN
            p[f"W2_{l}"] = rng.standard_normal((d, d_ff)) * s_ffn
            p[f"b2f_{l}"] = np.zeros(d)                                  # biais sortie FFN
        self.params = {k: v.astype(dtype) for k, v in p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.vv = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.mask = np.triu(np.ones((T, T), bool), k=1)                  # j > i masqué

    # ------------------------------------------------------------------ forward

    def forward(self, X, W1s):
        """X (B,T) codes ; W1s : liste de L matrices (d_ff,d) [fournisseur]. Retourne (perte, cache)."""
        B, T = X.shape
        E, P = self.params["E"], self.params["P"]
        x = E[X] + P[:T]
        cache = {"X": X, "layers": []}
        H, dh = self.H, self.dh
        for l in range(self.L):
            c = {"x_in": x}
            h1, c["xh1"], c["rs1"] = ln_forward(x, self.params[f"g1_{l}"], self.params[f"b1_{l}"])
            c["h1"] = h1
            q = h1 @ self.params[f"Wq_{l}"]
            k = h1 @ self.params[f"Wk_{l}"]
            v = h1 @ self.params[f"Wv_{l}"]
            q_ = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k_ = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            v_ = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            scores = q_ @ k_.transpose(0, 1, 3, 2) / np.sqrt(dh)
            scores[..., self.mask] = -1e30
            A = softmax(scores)
            o = (A @ v_).transpose(0, 2, 1, 3).reshape(B, T, self.d)
            c.update(A=A, q_=q_, k_=k_, v_=v_, o=o)
            x = x + o @ self.params[f"Wp_{l}"]
            c["x_mid"] = x
            h2, c["xh2"], c["rs2"] = ln_forward(x, self.params[f"g2_{l}"], self.params[f"b2_{l}"])
            c["h2"] = h2
            z = relu(h2 @ W1s[l].T + self.params[f"b1f_{l}"])
            c["z"] = z
            f = z @ self.params[f"W2_{l}"].T + self.params[f"b2f_{l}"]
            x = x + f
            cache["layers"].append(c)
        hf, cache["xhf"], cache["rsf"] = ln_forward(x, self.params["gf"], self.params["bf"])
        cache["hf"] = hf
        logits = hf @ self.params["E"].T                               # (B,T,V)
        Pp = softmax(logits.astype(np.float64))
        y = X[:, 1:]
        ib = np.arange(B)[:, None]
        it = np.arange(T - 1)[None, :]
        loss = -np.log(Pp[ib, it, y] + 1e-12).mean()
        cache["Pp"], cache["y"] = Pp, y
        return float(loss), cache

    def eval_loss(self, X, W1s):
        """Perte moyenne par lot complet (sans cache) : X (N,T)."""
        n = 32
        losses = []
        for i in range(0, X.shape[0], n):
            losses.append(self.forward(X[i:i + n], W1s)[0])
        return float(np.mean(losses))

    # ----------------------------------------------------------------- backward

    def backward(self, cache, W1s):
        """Retourne (grads des paramètres du modèle, Gs : ∂L/∂W1ℓ par couche)."""
        X = cache["X"]
        B, T = X.shape
        H, dh = self.H, self.dh
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        Gs = [None] * self.L

        dlogits = cache["Pp"].copy()
        ib = np.arange(B)[:, None]
        it = np.arange(T - 1)[None, :]
        dlogits[ib, it, cache["y"]] -= 1.0
        dlogits[:, -1, :] = 0.0                      # position T-1 : aucune cible dans la perte
        dlogits /= (B * (T - 1))
        grads["E"] += np.einsum("btv,btd->vd", dlogits, cache["hf"]).astype(self.dtype)
        dhf = (dlogits @ self.params["E"]).astype(self.dtype)
        dx, dgf, dbf = ln_backward(dhf, cache["xhf"].astype(self.dtype),
                                   cache["rsf"].astype(self.dtype), self.params["gf"])
        grads["gf"], grads["bf"] = dgf, dbf

        for l in reversed(range(self.L)):
            c = cache["layers"][l]
            # ---- FFN : x_out = x_mid + f
            df = dx                                                    # chemin résiduel
            grads[f"W2_{l}"] += np.einsum("bti,btj->ij", df, c["z"]).astype(self.dtype)
            grads[f"b2f_{l}"] += df.sum(axis=(0, 1))
            dz = (df @ self.params[f"W2_{l}"]) * (c["z"] > 0)
            grads[f"b1f_{l}"] += dz.sum(axis=(0, 1))
            Gs[l] = np.einsum("btf,btj->fj", dz, c["h2"]).astype(self.dtype)  # ∂L/∂W1_l
            dh2 = dz @ W1s[l]
            d_from_ln2, dg2, db2 = ln_backward(dh2, c["xh2"].astype(self.dtype),
                                               c["rs2"].astype(self.dtype), self.params[f"g2_{l}"])
            grads[f"g2_{l}"] += dg2; grads[f"b2_{l}"] += db2
            dx_mid = df + d_from_ln2                                   # ∂L/∂x_mid
            # ---- attention : x_mid = x_in + attn_out,  attn_out = o @ Wp
            d_attn = dx_mid                                            # ∂L/∂attn_out (B,T,d)
            grads[f"Wp_{l}"] += np.einsum("bti,btj->ij", c["o"], d_attn).astype(self.dtype)
            do_cat = d_attn @ self.params[f"Wp_{l}"].T                 # ∂L/∂o (B,T,d)
            do = do_cat.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            A, q_, k_, v_ = c["A"], c["q_"], c["k_"], c["v_"]
            dA = do @ v_.transpose(0, 1, 3, 2)
            dv_ = A.transpose(0, 1, 3, 2) @ do
            dscores = A * (dA - (dA * A).sum(-1, keepdims=True))
            dq_ = (dscores @ k_) / np.sqrt(dh)
            dk_ = (dscores.transpose(0, 1, 3, 2) @ q_) / np.sqrt(dh)
            dq = dq_.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dk = dk_.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dv = dv_.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            for nm, dd in (("q", dq), ("k", dk), ("v", dv)):
                grads[f"W{nm}_{l}"] += np.einsum("bti,btj->ij", c["h1"], dd).astype(self.dtype)
            # ∂L/∂h1 = rétropropagation à travers les projections q, k, v
            dh1 = (dq @ self.params[f"Wq_{l}"].T
                   + dk @ self.params[f"Wk_{l}"].T
                   + dv @ self.params[f"Wv_{l}"].T)
            d_from_ln1, dg1, db1 = ln_backward(dh1, c["xh1"].astype(self.dtype),
                                               c["rs1"].astype(self.dtype), self.params[f"g1_{l}"])
            grads[f"g1_{l}"] += dg1; grads[f"b1_{l}"] += db1
            dx = dx_mid + d_from_ln1                                   # ∂L/∂x_in

        np.add.at(grads["E"], X, dx.astype(self.dtype))
        grads["P"][:T] += dx.sum(axis=0)
        return grads, Gs

    # --------------------------------------------------------------------- Adam

    def step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        c1 = 1 - b1 ** self.t
        c2 = 1 - b2 ** self.t
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.vv[k] = b2 * self.vv[k] + (1 - b2) * g * g
            self.params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.vv[k] / c2) + eps)
