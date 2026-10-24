# -*- coding: utf-8 -*-
"""Figure d'illustration : un seul g, trois couches.

g(u) = u + Σₖ aₖ φₖ(u), K = 4 gaussiennes, u ∈ ℝ². Chaque couche ℓ lit la jacobienne de g en son point uℓ :
    Wℓ = J_g(uℓ) = I + Σₖ aₖ ∇φₖ(uℓ)ᵀ      (matrice 2×2 = 4 paramètres par couche)
La ligne i de Wℓ est le gradient de gᵢ en uℓ : on la voit comme la pente d'un plan tangent à la surface
hᵢ(u) = gᵢ(u) − uᵢ = Σₖ aₖ,ᵢ φₖ(u). Les trois plans tangents d'une même surface sont donc liés par la
continuité de g — c'est exactement la contrainte d'intégrabilité.

Sortie : figures/gaussiennes_3d.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genjac import W_field

# ------------------------------------------------------------------ le générateur (K = 4, d = 2)
K, d = 4, 2
C = np.array([[-1.1, 0.9], [0.9, 1.0], [-0.5, -1.0], [1.3, -0.7]])
R = np.array([0.75, 0.70, 0.80, 0.65])
A = np.array([[1.0, -0.6], [-0.8, 0.9], [0.7, 0.8], [-1.0, -0.7]])
THETA = np.concatenate([C.ravel(), np.log(R), A.ravel()])
U_LAYERS = np.array([[-1.4, 0.1], [0.0, 0.0], [1.4, 0.2]])
W_LAYERS = W_field(THETA, U_LAYERS, K, d)                       # (3, 2, 2) : Wℓ = J_g(uℓ)


def phi(u):
    """(…, K) valeurs des gaussiennes."""
    dv = u[..., None, :] - C
    return np.exp(-np.sum(dv ** 2, -1) / (2 * R ** 2))


def h(u, i):
    """hᵢ(u) = gᵢ(u) − uᵢ = Σₖ aₖ,ᵢ φₖ(u)."""
    return phi(u) @ A[:, i]


# ------------------------------------------------------------------ style
BG = "#0e1320"
PANEL = "#151b2d"
INK = "#e8ecf4"
MUTED = "#8a93ab"
LAYER_COL = ["#ff7a59", "#ffd166", "#4fd1c5"]            # couche 1, 2, 3
CMAP = colors.LinearSegmentedColormap.from_list(
    "dune", ["#1f5fa8", "#4f97cf", "#a9d0e8", "#ece8e4", "#f4c19f", "#e9855a", "#b8432f"])
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})


def style_axes(ax):
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"].update(color=(1, 1, 1, 0.07), linewidth=0.6)
        axis.line.set_color((1, 1, 1, 0.25))
        axis.set_tick_params(colors=MUTED, labelsize=7, pad=-2)
    ax.set_xlabel("u₁", fontsize=9, labelpad=-6)
    ax.set_ylabel("u₂", fontsize=9, labelpad=-6)


def surface_panel(ax, i, title):
    xs = np.linspace(-2.3, 2.3, 170)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X, Y], -1)
    Z = h(P, i)
    zmax = max(abs(Z.min()), abs(Z.max())) * 1.0
    norm = colors.TwoSlopeNorm(vmin=-zmax, vcenter=0, vmax=zmax)
    rgb = CMAP(norm(Z))
    shade = matplotlib.colors.LightSource(azdeg=235, altdeg=55).shade_rgb(rgb[..., :3], Z, fraction=0.8, blend_mode="soft", vert_exag=1.0, dx=xs[1] - xs[0], dy=xs[1] - xs[0])
    ax.plot_surface(X, Y, Z, facecolors=shade, rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=False, alpha=0.86, zorder=1)
    floor = -zmax * 1.25
    ax.contourf(X, Y, Z, zdir="z", offset=floor, levels=np.linspace(-zmax, zmax, 15), cmap=CMAP, norm=norm,
                alpha=0.35)
    ax.contour(X, Y, Z, zdir="z", offset=floor, levels=np.linspace(-zmax, zmax, 15), colors=[(1, 1, 1, 0.18)],
               linewidths=0.4)
    # empreinte des 4 gaussiennes au sol : centre + cercle de rayon Rₖ
    th = np.linspace(0, 2 * np.pi, 120)
    for k in range(K):
        ax.plot(C[k, 0] + R[k] * np.cos(th), C[k, 1] + R[k] * np.sin(th), floor + 0 * th,
                color=(1, 1, 1, 0.55), lw=0.8, ls=(0, (3, 3)))
        ax.scatter([C[k, 0]], [C[k, 1]], [floor], s=16, color="white", alpha=0.9, depthshade=False)
        ax.text(C[k, 0], C[k, 1], floor, f" {k + 1}", color=INK, fontsize=7, alpha=0.8)
    # points de couche, plans tangents
    for l in range(3):
        u = U_LAYERS[l]
        z0 = float(h(u, i))
        gx, gy = W_LAYERS[l][i] - np.eye(2)[i]                  # (∂hᵢ/∂u₁, ∂hᵢ/∂u₂) = ligne i de W − I
        s = 0.45
        quad = np.array([[u[0] + dx, u[1] + dy, z0 + gx * dx + gy * dy]
                         for dx, dy in [(-s, -s), (s, -s), (s, s), (-s, s)]])
        pc = Poly3DCollection([quad], facecolor=colors.to_rgba(LAYER_COL[l], 0.97),
                              edgecolor="white", linewidths=1.6)
        pc.set_zsort("max"); pc.set_sort_zpos(z0 + 1.0)
        ax.add_collection3d(pc)
        ax.plot([u[0], u[0]], [u[1], u[1]], [floor, z0], color=LAYER_COL[l], lw=1.0, ls=(0, (2, 2)), alpha=0.9)
        ax.scatter([u[0]], [u[1]], [floor], s=34, color=LAYER_COL[l], alpha=0.95, depthshade=False,
                   edgecolors="white", linewidths=0.6)
        ax.scatter([u[0]], [u[1]], [z0], s=210, color=LAYER_COL[l], edgecolors="white", linewidths=1.4,
                   depthshade=False, zorder=10)
        ax.text(u[0], u[1], z0 + zmax * 0.42, f"u{'₁₂₃'[l]}", color=LAYER_COL[l], fontsize=12,
                fontweight="bold", ha="center", zorder=11)
    ax.set_zlim(floor, zmax * 1.25)
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    ax.set_zticks([])
    ax.set_title(title, color=INK, fontsize=12, pad=0, y=0.97)
    ax.view_init(elev=40, azim=-62)
    ax.set_box_aspect((1, 1, 0.6))
    style_axes(ax)


def matrix_card(ax, l):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=PANEL, ec=LAYER_COL[l], lw=1.6, transform=ax.transAxes))
    ax.text(0.5, 0.88, f"couche {l + 1}", ha="center", va="center", fontsize=11.5, color=LAYER_COL[l],
            fontweight="bold")
    ax.text(0.5, 0.775, f"W{'₁₂₃'[l]} = J_g(u{'₁₂₃'[l]})", ha="center", va="center", fontsize=9.5, color=MUTED)
    W = W_LAYERS[l]
    norm = colors.TwoSlopeNorm(vmin=-2.0, vcenter=0, vmax=2.0)
    cell = 0.25
    x0, y0 = 0.5 - cell, 0.12
    for r in range(2):
        for c in range(2):
            v = W[r, c]
            col = plt.cm.RdBu_r(norm(v))
            ax.add_patch(FancyBboxPatch((x0 + c * cell + 0.012, y0 + (1 - r) * cell * 1.15 + 0.012),
                                        cell - 0.024, cell * 1.15 - 0.024,
                                        boxstyle="round,pad=0,rounding_size=0.03", fc=colors.to_rgba(col, 0.85),
                                        ec="none", transform=ax.transAxes))
            ax.text(x0 + (c + 0.5) * cell, y0 + (1 - r) * cell * 1.15 + cell * 0.575, f"{v:+.2f}",
                    ha="center", va="center", fontsize=12.5, color="white", fontweight="bold")


fig = plt.figure(figsize=(14.2, 9.4), facecolor=BG)
gs = GridSpec(2, 6, figure=fig, height_ratios=[2.35, 1.0], left=0.02, right=0.98, top=0.86, bottom=0.03,
              hspace=0.04, wspace=0.25)
ax1 = fig.add_subplot(gs[0, 0:3], projection="3d")
ax2 = fig.add_subplot(gs[0, 3:6], projection="3d")
surface_panel(ax1, 0, "ligne 1 de W   =   pente de la surface  h₁(u)")
surface_panel(ax2, 1, "ligne 2 de W   =   pente de la surface  h₂(u)")
for l in range(3):
    matrix_card(fig.add_subplot(gs[1, 2 * l:2 * l + 2]), l)

fig.text(0.5, 0.955, "Un seul g, trois couches", ha="center", fontsize=23, fontweight="bold", color=INK)
fig.text(0.5, 0.917, "Chaque couche lit la jacobienne de la même fonction g en son propre point :   "
         "Wℓ = J_g(uℓ) = I + Σₖ aₖ ∇φₖ(uℓ)ᵀ", ha="center", fontsize=11.5, color=MUTED)
fig.text(0.5, 0.884, "g = u + 4 gaussiennes (cercles blancs au sol, numérotés 1 à 4)  ·  chaque plan coloré est tangent "
         "à la surface au point de sa couche  ·  2×2 = 4 paramètres par couche",
         ha="center", fontsize=9.3, color=MUTED, alpha=0.9)
out = ROOT / "figures" / "gaussiennes_3d.png"
fig.savefig(out, dpi=200, facecolor=BG)
print("→", out)
