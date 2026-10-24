#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test d'échelle : matrices de poids = jacobiennes d'un générateur partagé.

Portage PyTorch du protocole Phase 3 (jacobian-weight-generator, NumPy pur) :
un transformer GPT-like niveau caractère dont la matrice W1 du FFN de chaque couche est
produite par un des cinq fournisseurs, à budget apparié :

  - gen     : Wℓ = M + Σₖ aₖ vₖ(uℓ)ᵀ, vₖ(u) = −(φₖ(u)/Rₖ²)(u − cₖ), φₖ gaussienne —
              un seul générateur g pour les L couches, Wℓ = J_g(uℓ) (famille INTÉGRABLE) ;
  - lora    : Wℓ = M + AℓBℓᵀ, rang r libre par couche (Hu et al. 2021) ;
  - hyp     : Wℓ = M + âℓŵℓᵀ, MLP(plongement de couche) → (â, ŵ) (sortie libre) ;
  - albert  : Wℓ = W partagé (Lan et al. 2020) ;
  - dense   : L matrices libres (référence haute, budget ≈ 3× le trio apparié).

À l'échelle (d = 128, d_ff = 512, L = 6 par défaut), les questions pré-enregistrées :
  Q1 (performance) : le générateur égale-t-il LoRA et l'hypernetwork à budget apparié,
     et réduit-il l'écart au dense ? GO si Δ(gen − lora) et Δ(gen − hyp) appariés par
     graine sont < 0.005 nats (test t apparié p ≥ 0.05), sur un corpus réel.
  Q2 (mécanistique, la vraie question d'échelle) : à cette échelle, le générateur
     DIFFÉRENCIE-t-il ses couches (Phase 3 : quasi-ALBERT, distances 0,6–15 %) ?
     Mesure : distances relatives ‖Wℓ − W₀‖/‖W₀‖ et rangs effectifs.
  NO-GO consigné avec la même rigueur qu'un succès.

Choix de protocole :
  * init « vivante » du générateur, indépendante de d (`c_init=live`, défaut) : l'init naïve
    (centres ~ 0,8·N(0,1) par coordonnée) donnait φ_max ≈ 1e-9 en d = 128 ⇒ W ≈ M, Q2 jouée d'avance ;
  * ablation `genfree` : même base gaussienne, amplitudes de rang 1 libres, SANS contrainte d'intégrabilité ;
  * critères : équivalence par TOST (IC 90 % de Δ apparié dans ±0,005 nats, ≥ 5 graines), supériorité
    à ALBERT (Δ < 0, p < 0,05) et condition préalable de discrimination (dense doit battre ALBERT) ;
  * `selftest` vérifie que les gaussiennes sont vivantes à l'init (φ_max > 0,1) en d = 128.

Écarts du protocole d'échelle (identiques pour toutes les méthodes) : warmup linéaire
300 pas + clipping de gradient 1.0. Optimiseur Adam sans weight decay, lr cosinus
3e-3 → 3e-4, comme en Phase 3.

Utilisation :
    # 1. auto-diagnostic (budgets appariés, gradients autograd vs différences finies)
    python jwg_scale.py selftest

    # 2. un corpus réel (recommandé) — par ex. tiny Shakespeare :
    curl -s https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o shakespeare.txt
    #    (sinon : corpus synthétique zipf intégré, ~2 M caractères)

    # 3. un run
    python jwg_scale.py train --method gen --seed 0 --corpus shakespeare.txt
    # 4. le banc complet + stats appariées
    python jwg_scale.py bench --methods gen lora hyp albert dense --seeds 0 1 2 --corpus shakespeare.txt

    # La question Q2 se sonde aussi via --u-init line|spread|random
    # (line = Phase 3 ; spread = points de couche orthogonaux forcés au départ).

Résultats : results_scale/{tag}_{method}_s{seed}.json (+ _W1.pt), puis
results_scale/summary_{tag}.json (tableau + tests t appariés imprimés).

Prérequis : torch, numpy. matplotlib optionnel (--plot). GPU conseillé (un run
défaut ≈ 6 000 pas × 4 096 jetons ; ~10 min sur RTX 3060, ~25 min sur MPS/CPU moderne,
plusieurs heures sur CPU seul — réduire alors --d 64 --d-ff 256 --L 4 --T 128 --B 16).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_scale"

CONS = "bcdfglmnprstv"
VOW = "aeiou"


# ============================================================ corpus


def make_zipf_text(n_chars, seed=0, n_words=800):
    """Corpus synthétique zipfien (portage de la tâche A de la Phase 3, agrandi)."""
    rng = np.random.default_rng(seed)
    words = []
    for _ in range(n_words):
        nsyl = rng.integers(1, 5)
        w = "".join(rng.choice(list(CONS)) + rng.choice(list(VOW)) for _ in range(nsyl))
        if rng.random() < 0.3:
            w += rng.choice(list(CONS))
        words.append(w)
    weights = 1.0 / (np.arange(len(words)) + 5.0)
    weights /= weights.sum()
    out, n = [], 0
    while n < n_chars:
        ns = rng.integers(5, 14)
        sent = []
        for i in range(ns):
            sent.append(words[rng.choice(len(words), p=weights)])
            if i < ns - 1 and rng.random() < 0.08:
                sent[-1] += ","
        out.append(" ".join(sent) + ".")
        n += ns * 7
    return " ".join(out)[:n_chars]


class CharData:
    """Texte → codes, avec split train/val, fenêtres d'éval et corpus bruité (10 % permutés)."""

    def __init__(self, text=None, codes=None, vocab=None):
        if text is not None:
            self.vocab = sorted(set(text))
            code = {c: i for i, c in enumerate(self.vocab)}
            self.data = np.array([code[c] for c in text], dtype=np.int64)
        else:
            self.vocab, self.data = list(vocab), np.asarray(codes, dtype=np.int64)
        self.V = len(self.vocab)
        self.code = {c: i for i, c in enumerate(self.vocab)}

    def split(self, frac=0.9):
        k = int(len(self.data) * frac)
        return CharData(codes=self.data[:k], vocab=self.vocab), \
               CharData(codes=self.data[k:], vocab=self.vocab)

    def sample_batch(self, B, T, rng):
        ix = rng.integers(0, len(self.data) - T - 1, size=B)
        return torch.from_numpy(np.stack([self.data[i:i + T] for i in ix]))

    def eval_windows(self, T, n_windows):
        ix = np.linspace(0, len(self.data) - T - 1, max(1, n_windows)).astype(int)
        return torch.from_numpy(np.stack([self.data[i:i + T] for i in ix]))

    def perturbed(self, frac=0.1, seed=777):
        """Phase 3 : 10 % des caractères permutés (mesure de robustesse au bruit)."""
        rng = np.random.default_rng(seed)
        data = self.data.copy()
        k = int(len(data) * frac)
        ix = rng.choice(len(data), size=k, replace=False)
        vals = data[ix].copy()
        rng.shuffle(vals)
        data[ix] = vals
        return CharData(codes=data, vocab=self.vocab)


def load_corpus(args):
    if args.corpus:
        text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
        if args.limit_chars:
            text = text[: args.limit_chars]
        tag = Path(args.corpus).stem
    else:
        text = make_zipf_text(args.synth_chars, seed=0)
        tag = "zipf"
    return CharData(text), tag


# ============================================================ fournisseurs de W1


def _gaussian_geometry(d, L, K, u_init, c_init, gen):
    """Points de couche U (L,d), centres C (K,d), log-rayons (K,).
    c_init = "naive" : init naïve (ablation) — centres ~ 0,8·N(0,1) par coordonnée, donc
        ‖u − c‖² ~ 0,64·d et φ ~ exp(−0,32·d/R²) : 1e-9 en d = 128 (gaussiennes mortes).
    c_init = "live" (défaut) : géométrie fixée en NORME, indépendante de d — points de couche espacés
        de 0,5 (line) / orthogonaux de norme 0,75 (spread) / aléatoires de norme ~0,5 (random), bruit de
        norme 0,05 ; centres posés près des points de couche (bruit de norme 0,3) ; R ~ U(0,5 ; 1,5)."""
    def rn(*shape, std=1.0):
        return torch.randn(*shape, generator=gen) * std

    if c_init == "naive":
        C = rn(K, d, std=0.8)
        logR = torch.log(torch.empty(K).uniform_(0.3, 1.5, generator=gen))
        noise = rn(L, d, std=0.05)
        if u_init == "line":
            base = torch.tensor([(l - (L - 1) / 2) * 0.5 for l in range(L)])
            U = base[:, None].repeat(1, d) + noise
        elif u_init == "spread":
            if d >= L:
                q, _ = torch.linalg.qr(torch.randn(d, L, generator=gen))
                U = 0.75 * q.T + noise
            else:
                U = 0.75 * F.normalize(rn(L, d), dim=1) + noise
        elif u_init == "random":
            U = rn(L, d, std=0.5)
        else:
            raise ValueError(u_init)
        return nn.Parameter(U), nn.Parameter(C), nn.Parameter(logR)
    if c_init != "live":
        raise ValueError(c_init)
    sd = 1.0 / math.sqrt(d)
    noise = rn(L, d, std=0.05 * sd)
    if u_init == "line":
        e = F.normalize(rn(d), dim=0)
        U = torch.stack([(l - (L - 1) / 2) * 0.5 * e for l in range(L)]) + noise
    elif u_init == "spread":
        if d >= L:
            q, _ = torch.linalg.qr(torch.randn(d, L, generator=gen))
            U = 0.75 * q.T + noise
        else:
            U = 0.75 * F.normalize(rn(L, d), dim=1) + noise
    elif u_init == "random":
        U = rn(L, d, std=0.5 * sd)
    else:
        raise ValueError(u_init)
    idx = torch.arange(K) % L
    C = U[idx].clone() + rn(K, d, std=0.3 * sd)
    logR = torch.log(torch.empty(K).uniform_(0.5, 1.5, generator=gen))
    return nn.Parameter(U), nn.Parameter(C), nn.Parameter(logR)


class GenProvider(nn.Module):
    """Wℓ = M + Σₖ aₖ vₖ(uℓ)ᵀ — le générateur jacobien (autograd fait la rétroprop)."""

    def __init__(self, d, d_ff, L, K, u_init="line", gen=None, c_init="live"):
        super().__init__()
        std_M = math.sqrt(2.0 / (d + d_ff))
        self.M = nn.Parameter(torch.randn(d_ff, d, generator=gen) * std_M)
        self.A = nn.Parameter(torch.randn(K, d_ff, generator=gen) * 0.02)
        self.U, self.C, self.logR = _gaussian_geometry(d, L, K, u_init, c_init, gen)

    def w1(self):
        R = torch.exp(self.logR)                     # (K,)
        dvec = self.U[:, None, :] - self.C[None, :, :]  # (L,K,d)
        s = (dvec * dvec).sum(-1)                     # (L,K)
        phi = torch.exp(-s / (2 * R * R)[None, :])    # (L,K)
        v = -(phi / (R * R))[..., None] * dvec        # (L,K,d)
        return self.M[None] + torch.einsum("kf,lkd->lfd", self.A, v)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def diag(self):
        R = torch.exp(self.logR)
        s = ((self.U[:, None, :] - self.C[None]) ** 2).sum(-1)
        phi = torch.exp(-s / (2 * R * R)[None])
        return {"phi_max": float(phi.max()), "phi_median": float(phi.median()),
                "n_active_gt_0.01": int((phi > 0.01).sum())}


class GenFreeProvider(nn.Module):
    """ABLATION non intégrable : Wℓ = M + Σₖ φₖ(uℓ)·aₖbₖᵀ — même base gaussienne et même géométrie de
    départ que le générateur, mais amplitudes de rang 1 LIBRES (le champ n'est plus un champ de jacobiennes).
    Isole l'apport de la contrainte d'intégrabilité."""

    def __init__(self, d, d_ff, L, K, u_init="line", gen=None, c_init="live"):
        super().__init__()
        self.M = nn.Parameter(torch.randn(d_ff, d, generator=gen) * math.sqrt(2.0 / (d + d_ff)))
        self.A = nn.Parameter(torch.randn(K, d_ff, generator=gen) * 0.02)
        self.B = nn.Parameter(torch.randn(K, d, generator=gen) * (0.3 / math.sqrt(d)))
        self.U, self.C, self.logR = _gaussian_geometry(d, L, K, u_init, c_init, gen)

    def w1(self):
        R = torch.exp(self.logR)
        s = ((self.U[:, None, :] - self.C[None]) ** 2).sum(-1)
        phi = torch.exp(-s / (2 * R * R)[None])                     # (L,K)
        return self.M[None] + torch.einsum("lk,kf,kd->lfd", phi, self.A, self.B)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class LoraProvider(nn.Module):
    """Wℓ = M + AℓBℓᵀ, rang r, facteurs libres par couche (B nul au départ)."""

    def __init__(self, d, d_ff, L, r, gen=None):
        super().__init__()
        self.r = r
        self.M = nn.Parameter(torch.randn(d_ff, d, generator=gen)
                              * math.sqrt(2.0 / (d + d_ff)))
        self.A = nn.Parameter(torch.randn(L, d_ff, r, generator=gen) * 0.1)
        self.B = nn.Parameter(torch.zeros(L, d, r))

    def w1(self):
        return self.M[None] + torch.einsum("lfr,ldr->lfd", self.A, self.B)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class HypProvider(nn.Module):
    """Wℓ = M + âℓŵℓᵀ avec (â,ŵ) = MLP(plongement de couche) — sortie libre de rang 1."""

    def __init__(self, d, d_ff, L, e_dim, Hh, gen=None):
        super().__init__()
        self.M = nn.Parameter(torch.randn(d_ff, d, generator=gen)
                              * math.sqrt(2.0 / (d + d_ff)))
        self.Emb = nn.Parameter(torch.randn(L, e_dim, generator=gen) * 0.5)
        self.W1h = nn.Parameter(torch.randn(e_dim, Hh, generator=gen)
                                * math.sqrt(1.0 / e_dim))
        self.b1h = nn.Parameter(torch.zeros(Hh))
        self.Wa = nn.Parameter(torch.randn(Hh, d_ff, generator=gen) * 0.02)
        self.ba = nn.Parameter(torch.zeros(d_ff))
        self.Ww = nn.Parameter(torch.randn(Hh, d, generator=gen) * 0.02)
        self.bw = nn.Parameter(torch.zeros(d))

    def w1(self):
        h = torch.tanh(self.Emb @ self.W1h + self.b1h)      # (L,Hh)
        a = h @ self.Wa + self.ba                            # (L,d_ff)
        w = h @ self.Ww + self.bw                            # (L,d)
        return self.M[None] + a[..., None] * w[..., None, :]

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class AlbertProvider(nn.Module):
    """Wℓ = W partagé (une seule matrice libre)."""

    def __init__(self, d, d_ff, L, gen=None):
        super().__init__()
        self.L = L
        self.W = nn.Parameter(torch.randn(d_ff, d, generator=gen)
                              * math.sqrt(2.0 / (d + d_ff)))

    def w1(self):
        return self.W[None].expand(self.L, -1, -1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class DenseProvider(nn.Module):
    """L matrices libres — référence haute."""

    def __init__(self, d, d_ff, L, gen=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, d_ff, d, generator=gen)
                              * math.sqrt(2.0 / (d + d_ff)))

    def w1(self):
        return self.W

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def solve_budgets(d, d_ff, L, target, e_dim=16):
    """Choisit (K, r, Hh) pour appariement ±2 % du budget cible du trio gen/lora/hyp."""
    M = d * d_ff
    U = L * d
    K = max(1, round((target - M - U) / (d + d_ff + 1)))
    r = max(1, round((target - M) / (L * (d + d_ff))))
    Hh = max(1, round((target - M - L * e_dim - d_ff - d)
                      / (e_dim + 1 + d_ff + d)))
    Kf = max(1, round((target - M - U) / (d_ff + 2 * d + 1)))        # genfree : K'(d_ff + 2d + 1)
    budgets = {
        "gen": M + U + K * (d + d_ff + 1),
        "genfree": M + U + Kf * (d_ff + 2 * d + 1),
        "gen_naive": M + U + K * (d + d_ff + 1),
        "lora": M + L * r * (d + d_ff),
        "hyp": M + L * e_dim + e_dim * Hh + Hh + Hh * d_ff + d_ff + Hh * d + d,
        "albert": M,
        "dense": L * d * d_ff,
    }
    trio = [budgets["gen"], budgets["lora"], budgets["hyp"]]
    spread = (max(trio) - min(trio)) / min(trio)
    assert spread <= 0.02 + 1e-9, f"budgets non appariés : {trio} (écart {spread:.1%})"
    assert abs(budgets["genfree"] - budgets["gen"]) / budgets["gen"] <= 0.02 + 1e-9, "genfree non apparié"
    solve_budgets.Kf = Kf
    return K, r, Hh, budgets


def make_provider(method, d, d_ff, L, budget, u_init, gen):
    K, r, Hh, budgets = solve_budgets(d, d_ff, L, budget)
    if method == "gen":
        return GenProvider(d, d_ff, L, K, u_init=u_init, gen=gen, c_init="live"), budgets
    if method == "gen_naive":                 # init naïve : gaussiennes mortes en grande dimension
        return GenProvider(d, d_ff, L, K, u_init=u_init, gen=gen, c_init="naive"), budgets
    if method == "genfree":
        return GenFreeProvider(d, d_ff, L, solve_budgets.Kf, u_init=u_init, gen=gen), budgets
    if method == "lora":
        return LoraProvider(d, d_ff, L, r, gen=gen), budgets
    if method == "hyp":
        return HypProvider(d, d_ff, L, e_dim=16, Hh=Hh, gen=gen), budgets
    if method == "albert":
        return AlbertProvider(d, d_ff, L, gen=gen), budgets
    if method == "dense":
        return DenseProvider(d, d_ff, L, gen=gen), budgets
    raise ValueError(method)


# ============================================================ transformer


class Block(nn.Module):
    """Pre-LN : x + Attn(LN1(x)) ; puis x + FFN(LN2(x)) avec W1 fournie.

    FFN : z = relu(h2 @ W1ᵀ + b1f) ; x = x + z @ W2ᵀ + b2f  (W1 (d_ff, d) vient du fournisseur).
    """

    def __init__(self, d, d_ff, H, gen=None):
        super().__init__()
        sa, sf = 1.0 / math.sqrt(d), math.sqrt(2.0 / (d + d_ff))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq = nn.Parameter(torch.randn(d, d, generator=gen) * sa)
        self.Wk = nn.Parameter(torch.randn(d, d, generator=gen) * sa)
        self.Wv = nn.Parameter(torch.randn(d, d, generator=gen) * sa)
        self.Wp = nn.Parameter(torch.randn(d, d, generator=gen) * sa)
        self.W2 = nn.Parameter(torch.randn(d, d_ff, generator=gen) * sf)
        self.b1f = nn.Parameter(torch.zeros(d_ff))
        self.b2f = nn.Parameter(torch.zeros(d))
        self.H = H

    def forward(self, x, W1, mask):
        B, T, d = x.shape
        dh = d // self.H
        h = self.ln1(x)
        q = (h @ self.Wq).view(B, T, self.H, dh).transpose(1, 2)
        k = (h @ self.Wk).view(B, T, self.H, dh).transpose(1, 2)
        v = (h @ self.Wv).view(B, T, self.H, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + o @ self.Wp
        h2 = self.ln2(x)
        z = F.relu(h2 @ W1.t() + self.b1f)
        return x + z @ self.W2.t() + self.b2f


class ScaleLM(nn.Module):
    """Embeddings + positions, L blocs, LN finale, tête liée (poids partagés avec E)."""

    def __init__(self, V, d, d_ff, L, H, T, gen=None):
        super().__init__()
        self.E = nn.Parameter(torch.randn(V, d, generator=gen) * 0.1)
        self.P = nn.Parameter(torch.randn(T, d, generator=gen) * 0.02)
        self.blocks = nn.ModuleList([Block(d, d_ff, H, gen=gen) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(self, idx, W1s):
        x = self.E[idx] + self.P[: idx.shape[1]]
        for l, blk in enumerate(self.blocks):
            x = blk(x, W1s[l], self.mask[: idx.shape[1], : idx.shape[1]])
        return self.lnf(x) @ self.E.t()          # tête liée


# ============================================================ entraînement


def lr_at(step, total, lr0, lr1, warmup):
    if step < warmup:
        return lr0 * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_loss(model, provider, X, chunk=32):
    W1s = provider.w1()
    losses = []
    for i in range(0, X.shape[0], chunk):
        logits = model(X[i:i + chunk], W1s)
        x = X[i:i + chunk]
        losses.append(F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1)).item())
    return float(np.mean(losses))


def w1_diagnostics(W):
    """Distances relatives inter-couches + rangs effectifs (Q2, Phase 3)."""
    W = W.double().cpu()
    L = W.shape[0]
    n0 = float(torch.linalg.norm(W[0]))
    rel = [float(torch.linalg.norm(W[l] - W[0]) / n0) for l in range(1, L)]
    pair = [float(torch.linalg.norm(W[i] - W[j]) /
                  (0.5 * (torch.linalg.norm(W[i]) + torch.linalg.norm(W[j]))))
            for i in range(L) for j in range(i + 1, L)]
    eff = []
    for l in range(L):
        sv = torch.linalg.svdvals(W[l])
        s2 = sv ** 2
        eff.append(float(s2.sum() ** 2 / (s2 ** 2).sum()))
    return {"rel_dist_to_l0": rel,
            "mean_pairwise_rel_dist": float(np.mean(pair)),
            "eff_ranks": eff}


def run_one(args, method, seed, corpus, tag):
    dev = torch.device(args.device)
    train, val = corpus.split()
    val_noise = val.perturbed(0.1, seed=seed + 777)
    ginit = torch.Generator().manual_seed(1000 + seed)      # ordre Phase 3 : modèle puis fournisseur
    gdata = np.random.default_rng(2000 + seed)

    torch.manual_seed(seed)
    model = ScaleLM(train.V, args.d, args.d_ff, args.L, args.heads, args.T, gen=ginit)
    provider, budgets = make_provider(method, args.d, args.d_ff, args.L,
                                      args.w1_budget, args.u_init, gen=ginit)
    model, provider = model.to(dev), provider.to(dev)
    all_params = list(model.parameters()) + list(provider.parameters())
    opt = torch.optim.Adam(all_params, lr=args.lr)
    Xval = val.eval_windows(args.T, args.val_windows).to(dev)
    Xval_n = val_noise.eval_windows(args.T, args.val_windows).to(dev)
    amp = args.amp and dev.type == "cuda"

    curve, t0 = [], time.time()
    for step in range(args.steps):
        X = train.sample_batch(args.B, args.T, gdata).to(dev)
        lr = lr_at(step, args.steps, args.lr, args.lr_final, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        W1s = provider.w1()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(X, W1s)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   X[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == 0:
            curve.append({"step": step + 1, "train": float(loss.item()),
                          "val": eval_loss(model, provider, Xval)})
            print(f"  [{method}/s{seed}] step {step+1}/{args.steps} "
                  f"train {loss.item():.4f} val {curve[-1]['val']:.4f}", flush=True)

    W1s = provider.w1().detach()
    res = {
        "method": method, "seed": seed, "steps": args.steps, "tag": tag,
        "ff1_params": provider.n_params,
        "budgets": budgets,
        "model_params": sum(p.numel() for p in model.parameters()),
        "final_train": eval_loss(model, provider,
                                 train.eval_windows(args.T, 64).to(dev)),
        "final_val": eval_loss(model, provider, Xval),
        "final_val_noise": eval_loss(model, provider, Xval_n),
        "best_val": min(c["val"] for c in curve),
        "w1_diag": w1_diagnostics(W1s),
        "u_init": args.u_init if method.startswith("gen") else None,
        "gen_diag": provider.diag() if hasattr(provider, "diag") else None,
        "curve": curve,
        "time_s": round(time.time() - t0, 1),
        "config": {k: getattr(args, k) for k in
                   ["d", "d_ff", "L", "heads", "T", "B", "lr", "lr_final", "warmup"]},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}_{method}_s{seed}.json").write_text(
        json.dumps(res, ensure_ascii=False))
    torch.save(W1s.cpu(), OUT / f"{tag}_{method}_s{seed}_W1.pt")
    print(f"[done] {tag}/{method}/s{seed}: val={res['final_val']:.4f} "
          f"noise={res['final_val_noise']:.4f} ff1={provider.n_params} "
          f"({res['time_s']}s)")
    return res


# ============================================================ statistiques


def _betacf(a, b, x):
    """Fraction continue de Lentz pour la bêta incomplète (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def paired_t(x, y):
    """Test t apparié bilatéral (implémentation numpy, sans scipy)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m, s = float(d.mean()), float(d.std(ddof=1))
    if s == 0:
        return 0.0, 1.0
    t = m / (s / math.sqrt(n))
    df = n - 1
    p = _betainc(df / 2.0, 0.5, df / (df + t * t))
    return t, p


MARGIN = 0.005          # marge d'équivalence pré-enregistrée (nats)


def t_crit(df, level=0.90):
    """Quantile bilatéral t_{(1+level)/2, df} par bissection sur la p-value exacte (sans scipy)."""
    alpha = 1 - level
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = _betainc(df / 2.0, 0.5, df / (df + mid * mid))
        lo, hi = (mid, hi) if p > alpha else (lo, mid)
    return 0.5 * (lo + hi)


def tost(x, y, margin=MARGIN):
    """Équivalence par IC 90 % du Δ apparié (TOST à 5 %) : équivalents ⇔ IC ⊂ (−margin, +margin)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    tc = t_crit(n - 1)
    lo, hi = m - tc * se, m + tc * se
    return {"delta": m, "ci90": [lo, hi], "equivalent": bool(lo > -margin and hi < margin), "n": n}


def verdict_phase4(by, stats):
    """Critères Phase 4 (fixés avant les runs) :
      0. DISCRIMINATION : dense bat ALBERT (Δ(dense − albert) < −0,005 et p < 0,05), sinon la tâche ne voit
         pas la paramétrisation de W₁ et Q1 est NON CONCLUANT quel que soit le reste ;
      Q1a. ÉQUIVALENCE : IC 90 % de Δ(gen − lora) et Δ(gen − hyp) dans ±0,005 (TOST), ≥ 5 graines ;
      Q1b. UTILITÉ : gen bat ALBERT (Δ < 0, p < 0,05) — sinon le partage simple suffit ;
      Q1c. INTÉGRABILITÉ : gen bat genfree (Δ < 0, p < 0,05) — sinon la contrainte ne sert à rien ;
      Q2. distance inter-couches moyenne du générateur > 5 % (init vivante vérifiée par gen_diag)."""
    lines, out = [], {}
    n = min((len(v) for v in by.values()), default=0)
    out["n_seeds_min"] = n
    if n < 5:
        lines.append(f"⚠ {n} graines < 5 : verdicts d'équivalence non recevables")
    if "dense" in by and "albert" in by:
        sd = {r["seed"]: r["final_val"] for r in by["dense"]}
        sa = {r["seed"]: r["final_val"] for r in by["albert"]}
        seeds = sorted(set(sd) & set(sa))
        if len(seeds) >= 2:
            t, p = paired_t([sd[k] for k in seeds], [sa[k] for k in seeds])
            dm = float(np.mean([sd[k] - sa[k] for k in seeds]))
            out["discrimine"] = bool(dm < -MARGIN and p < 0.05)
            lines.append(f"0. discrimination : Δ(dense−albert) = {dm:+.4f} (p={p:.3f}) → "
                         f"{'tâche discriminante' if out['discrimine'] else 'TÂCHE NON DISCRIMINANTE (Q1 non concluant)'}")
    for other in ("lora", "hyp"):
        k = f"gen_vs_{other}"
        if k in stats:
            tt = stats[k]["tost"]
            lines.append(f"Q1a. gen−{other}: Δ={tt['delta']:+.4f}, IC90=[{tt['ci90'][0]:+.4f}, {tt['ci90'][1]:+.4f}] → "
                         f"{'ÉQUIVALENT' if tt['equivalent'] else 'équivalence NON démontrée'}")
            out[f"equiv_{other}"] = tt["equivalent"]
    for other, tag in (("albert", "Q1b"), ("genfree", "Q1c")):
        k = f"gen_vs_{other}"
        if k in stats:
            st = stats[k]
            win = st["delta_moyen"] < 0 and st["p"] < 0.05
            lines.append(f"{tag}. gen−{other}: Δ={st['delta_moyen']:+.4f} (p={st['p']:.3f}) → "
                         f"{'gen MEILLEUR' if win else 'pas de gain démontré'}")
            out[f"beats_{other}"] = bool(win)
    if "gen" in by:
        dist = float(np.mean([r["w1_diag"]["mean_pairwise_rel_dist"] for r in by["gen"]]))
        phis = [r["gen_diag"]["phi_max"] for r in by["gen"] if r.get("gen_diag")]
        lines.append(f"Q2. distance inter-couches (gen) = {dist:.3f} "
                     f"({'> 5 % : couches différenciées' if dist > 0.05 else '≤ 5 % : quasi-partage'})"
                     + (f" ; φ_max final moyen = {np.mean(phis):.2f}" if phis else ""))
        out["q2_dist"] = dist
    out["lines"] = lines
    return out


def summarize(results, tag):
    by = {}
    for r in results:
        by.setdefault(r["method"], []).append(r)
    print(f"\n===== Résumé {tag} (val finale, moyenne ± écart-type) =====")
    table = {}
    for m, rs in sorted(by.items()):
        v = np.array([r["final_val"] for r in rs])
        table[m] = {"val_mean": float(v.mean()), "val_std": float(v.std(ddof=1)),
                    "n": len(v),
                    "dist_intercouches": float(np.mean(
                        [r["w1_diag"]["mean_pairwise_rel_dist"] for r in rs])),
                    "eff_rank_moyen": float(np.mean(
                        [np.mean(r["w1_diag"]["eff_ranks"]) for r in rs])),
                    "ff1_params": rs[0]["ff1_params"]}
        print(f"  {m:7s} {v.mean():.4f} ± {v.std(ddof=1):.4f} "
              f"(n={len(v)}, ff1={rs[0]['ff1_params']}, "
              f"dist. intercouches={table[m]['dist_intercouches']:.3f})")
    stats = {}
    if "gen" in by:
        for other in ("lora", "hyp", "albert", "dense", "genfree", "gen_naive"):
            if other in by:
                seeds = sorted(set(r["seed"] for r in by["gen"])
                               & set(r["seed"] for r in by[other]))
                if len(seeds) >= 2:
                    gv = [next(r["final_val"] for r in by["gen"] if r["seed"] == s)
                          for s in seeds]
                    ov = [next(r["final_val"] for r in by[other] if r["seed"] == s)
                          for s in seeds]
                    t, p = paired_t(gv, ov)
                    stats[f"gen_vs_{other}"] = {
                        "delta_moyen": float(np.mean(np.array(gv) - np.array(ov))),
                        "t": t, "p": p, "n": len(seeds), "tost": tost(gv, ov)}
                    print(f"  Δ apparié gen−{other}: {stats[f'gen_vs_{other}']['delta_moyen']:+.4f} "
                          f"(t={t:.2f}, p={p:.3f}, n={len(seeds)})")
    verdict = verdict_phase4(by, stats)
    for line in verdict["lines"]:
        print("  " + line)
    OUT.mkdir(exist_ok=True)
    (OUT / f"summary_{tag}.json").write_text(
        json.dumps({"table": table, "paired": stats, "verdict": verdict}, indent=1, ensure_ascii=False))
    print(f"→ {OUT / f'summary_{tag}.json'}")
    return table, stats


def plot_curves(results, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent — figure sautée")
        return
    by = {}
    for r in results:
        by.setdefault(r["method"], []).append(r)
    plt.figure(figsize=(6.4, 4.2))
    for m, rs in sorted(by.items()):
        steps = [c["step"] for c in rs[0]["curve"]]
        V = np.array([[c["val"] for c in r["curve"]] for r in rs])
        mean, std = V.mean(0), V.std(0, ddof=1) if len(rs) > 1 else np.zeros_like(V.mean(0))
        plt.plot(steps, mean, label=m)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Pas"), plt.ylabel("Perte de validation (nats/car.)")
    plt.title(tag), plt.legend(frameon=False)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(OUT / f"courbes_{tag}.{ext}", dpi=150)
    print(f"→ figures : {OUT}/courbes_{tag}.svg")


# ============================================================ selftest


def selftest(args):
    d, d_ff, L, target = 128, 512, 6, round(6 * 128 * 512 / 3)
    K, r, Hh, budgets = solve_budgets(d, d_ff, L, target)
    print(f"[1] budgets (d={d}, d_ff={d_ff}, L={L}, cible={target}) :")
    for k, v in budgets.items():
        print(f"      {k:7s} {v:>7d}")
    trio = [budgets["gen"], budgets["lora"], budgets["hyp"]]
    spread = (max(trio) - min(trio)) / min(trio)
    print(f"      trio apparié à {spread:.2%} — OK" if spread <= 0.02
          else "      ÉCHEC d'appariement")
    assert spread <= 0.02

    print("[1b] gaussiennes vivantes à l'init en d = 128 (garde-fou contre l'init naïve) :")
    for ui in ("line", "spread", "random"):
        for cinit in ("naive", "live"):
            gg = torch.Generator().manual_seed(0)
            pr = GenProvider(128, 512, 6, K, u_init=ui, gen=gg, c_init=cinit)
            ph = pr.diag()["phi_max"]
            print(f"      u_init={ui:6s} c_init={cinit:6s} φ_max = {ph:.1e}")
            if cinit == "live":
                assert ph > 0.1, "gaussiennes mortes à l'init (c_init=live)"
            else:
                assert ph < 1e-3, "l'init naïve devrait être morte en d=128"

    print("[2] gradients autograd du générateur vs différences finies (float64) :")
    g = torch.Generator().manual_seed(0)
    dd, dff, LL, KK = 8, 12, 3, 2
    prov = GenProvider(dd, dff, LL, KK, u_init="line", gen=g).double()
    Wt = torch.randn(LL, dff, dd, generator=g).double()
    max_rel = 0.0
    for name, p in prov.named_parameters():
        p.grad = None
        loss = (prov.w1() * Wt).sum()
        loss.backward()
        for _ in range(3):                          # 3 indices aléatoires par tenseur
            flat = p.view(-1)
            i = int(torch.randint(0, flat.numel(), (1,), generator=g))
            eps = 1e-6
            with torch.no_grad():
                orig = flat[i].item()
                flat[i] = orig + eps
                lp = float((prov.w1() * Wt).sum())
                flat[i] = orig - eps
                lm = float((prov.w1() * Wt).sum())
                flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            ag = p.grad.view(-1)[i].item()
            denom = max(abs(fd), abs(ag), 1e-8)
            max_rel = max(max_rel, abs(fd - ag) / denom)
    print(f"      erreur relative max : {max_rel:.2e} "
          f"({'OK' if max_rel < 1e-5 else 'ÉCHEC'})")
    assert max_rel < 1e-5

    print("[3] fumée : 60 pas en d=64 doivent faire baisser la perte")
    class A:
        pass
    a = A()
    a.d, a.d_ff, a.L, a.heads, a.T, a.B = 64, 256, 2, 2, 64, 8
    a.lr, a.lr_final, a.warmup, a.steps, a.eval_every, a.val_windows = 3e-3, 3e-4, 10, 60, 30, 16
    a.u_init, a.w1_budget = "line", round(2 * 64 * 256 / 3)
    a.amp, a.device = False, "cpu"
    corpus = CharData(make_zipf_text(20000, seed=0))
    r1 = run_one(a, "gen", 0, corpus, "smoke")
    r2 = run_one(a, "lora", 0, corpus, "smoke")
    ok = r1["curve"][-1]["train"] < r1["curve"][0]["train"] and \
         r2["curve"][-1]["train"] < r2["curve"][0]["train"]
    print(f"      gen {r1['curve'][0]['train']:.3f}→{r1['curve'][-1]['train']:.3f}, "
          f"lora {r2['curve'][0]['train']:.3f}→{r2['curve'][-1]['train']:.3f} "
          f"({'OK' if ok else 'ÉCHEC'})")
    assert ok
    print("\nSelftest complet : OK — lance le banc (bench) sur un vrai corpus.")


# ============================================================ CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--d", type=int, default=128)
        p.add_argument("--d-ff", type=int, default=512)
        p.add_argument("--L", type=int, default=6)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--T", type=int, default=128)
        p.add_argument("--B", type=int, default=32)
        p.add_argument("--steps", type=int, default=6000)
        p.add_argument("--warmup", type=int, default=300)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--lr-final", type=float, default=3e-4)
        p.add_argument("--eval-every", type=int, default=100)
        p.add_argument("--val-windows", type=int, default=256)
        p.add_argument("--w1-budget", type=int, default=0,
                       help="budget du trio gen/lora/hyp (défaut : dense/3)")
        p.add_argument("--u-init", choices=["line", "spread", "random"], default="line")
        p.add_argument("--corpus", type=str, default="",
                       help="fichier texte (défaut : corpus zipf synthétique)")
        p.add_argument("--limit-chars", type=int, default=0)
        p.add_argument("--synth-chars", type=int, default=2_000_000)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                       else ("mps" if getattr(torch.backends, "mps", None)
                             and torch.backends.mps.is_available() else "cpu"))
        p.add_argument("--amp", action="store_true",
                       help="autocast bf16 (CUDA uniquement)")

    p1 = sub.add_parser("train", help="un run (méthode × graine)")
    common(p1)
    p1.add_argument("--method", choices=["gen", "gen_naive", "genfree", "lora", "hyp", "albert", "dense"],
                    required=True)
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("bench", help="banc complet + stats appariées")
    common(p2)
    p2.add_argument("--methods", nargs="+", default=["gen", "lora", "hyp", "albert", "dense"],
                    choices=["gen", "gen_naive", "genfree", "lora", "hyp", "albert", "dense"])
    p2.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p2.add_argument("--resume", action="store_true",
                    help="réutilise les runs déjà sauvegardés")
    p2.add_argument("--plot", action="store_true")

    sub.add_parser("selftest", help="budgets + gradients autograd vs FD + fumée")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.cmd == "selftest":
        selftest(args)
        return

    if not args.w1_budget:
        args.w1_budget = round(args.L * args.d * args.d_ff / 3)
    assert args.d % args.heads == 0, "d doit être divisible par le nombre de têtes"
    corpus, tag = load_corpus(args)
    print(f"corpus « {tag} » : V={corpus.V} caractères={len(corpus.data):,} "
          f"device={args.device} "
          f"budget W1 trio={args.w1_budget:,} (dense={args.L*args.d*args.d_ff:,})")

    if args.cmd == "train":
        run_one(args, args.method, args.seed, corpus, tag)
    else:
        results = []
        for method in args.methods:
            for seed in args.seeds:
                f = OUT / f"{tag}_{method}_s{seed}.json"
                if args.resume and f.exists():
                    results.append(json.loads(f.read_text()))
                    print(f"[skip] {method}/s{seed} (déjà fait)")
                else:
                    results.append(run_one(args, method, seed, corpus, tag))
        summarize(results, tag)
        if args.plot:
            plot_curves(results, tag)


if __name__ == "__main__":
    main()
