"""Hypernetwork à sortie libre : h(u) = reshape(W₂·tanh(W₁u + b₁) + b₂) — baseline clé de la Phase 2
(même entrée u, SANS contrainte d'intégrabilité). Rétropropagation manuelle ; gradients vérifiés
contre les différences finies dans les tests."""
import numpy as np


class Hypernet:
    def __init__(self, d, H, rng):
        self.d, self.H = d, H
        # W₁ (H,d), b₁ (H,), W₂ (d²,H), b₂ (d²,)
        self.shapes = [((H, d), np.sqrt(1.0 / d)), ((H,), 0.1),
                       ((d * d, H), np.sqrt(1.0 / H)), ((d * d,), 0.1)]
        self.params = np.concatenate([(rng.standard_normal(s) * sc).reshape(-1)
                                      for s, sc in self.shapes])

    @property
    def n_params(self):
        return len(self.params)

    def _split(self):
        i = 0; out = []
        for (s, _), flat in zip(self.shapes, [None] * len(self.shapes)):
            n = int(np.prod(s)); out.append(self.params[i:i + n].reshape(s)); i += n
        return out  # W1, b1, W2, b2

    def W(self, u):
        """u (m,d) → (m,d,d)."""
        W1, b1, W2, b2 = self._split()
        h = np.tanh(u @ W1.T + b1)                     # (m,H)
        G = h @ W2.T + b2                              # (m,d²)
        return G.reshape(-1, self.d, self.d)

    def loss_grad(self, u, T):
        """L = Σ_i ½‖W(u_i) − T_i‖²_F et gradient par rapport à tous les paramètres."""
        W1, b1, W2, b2 = self._split()
        m = u.shape[0]
        h_pre = u @ W1.T + b1
        h = np.tanh(h_pre)
        G = h @ W2.T + b2
        W = G.reshape(m, self.d, self.d)
        E = W - T
        L = 0.5 * float(np.sum(E ** 2))
        dG = E.reshape(m, self.d * self.d)              # (m,d²)
        gW2 = dG.T @ h                                  # (d²,H)
        gb2 = dG.sum(axis=0)
        dh = dG @ W2                                    # (m,H)
        dh_pre = dh * (1 - h ** 2)
        gW1 = dh_pre.T @ u
        gb1 = dh_pre.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        flat = np.concatenate([g.reshape(-1) for g in grads])
        return L, flat

    def set(self, params):
        self.params = params
