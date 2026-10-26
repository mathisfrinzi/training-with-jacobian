# Générateur jacobien : entraîner des matrices comme jacobiennes d'une fonction

Projet clôturé le 2026-10-24. Résultat négatif, confirmé par la Phase 5.
Code : NumPy (`src/`), PyTorch pour le test d'échelle (`scale/`), 11 tests (`tests/`).

![Un seul g, trois couches](figures/gaussiennes_3d.png)

*Une fonction g = u + 4 gaussiennes, évaluée en trois points. Chaque couche obtient sa matrice 2×2 comme jacobienne de g en son point. Les plans colorés sont tangents à la surface hᵢ(u) = gᵢ(u) − uᵢ ; la ligne i de Wℓ est leur pente. Comme les trois plans sont tangents à la même surface, ils ne peuvent pas être choisis indépendamment : c'est la contrainte d'intégrabilité. (`python src/figure_3d.py`)*

## Motivation

L'idée : au lieu d'entraîner une matrice de poids coefficient par coefficient, la considérer comme la jacobienne d'une fonction g(u) = u + Σₖ aₖ φₖ(u) (φₖ gaussiennes) et entraîner g.

Évaluer g en un seul point n'apporte rien. Toute matrice est la jacobienne d'une fonction (g(u) = Wu suffit), et avec K gaussiennes on obtient W = I + ABᵀ, soit une perturbation de rang ≤ K : c'est une LoRA (Hu et al., 2021) reparamétrée. La Phase 1 le vérifie, avec en prime une optimisation moins fiable (88 à 93 % de convergence contre 100 % pour LoRA).

L'idée n'a d'intérêt que si le même g est évalué en plusieurs points u₁…u_m. Les matrices W(uᵢ) = J_g(uᵢ) forment alors un champ qui vérifie la condition d'intégrabilité ∂ₗ Wⱼₖ = ∂ₖ Wⱼₗ, contrainte que ni LoRA ni un hypernetwork classique n'imposent. La question du projet est de savoir si cette contrainte sert à quelque chose :

- généralisation à des points non vus (Phase 2) ;
- performance à budget égal quand un seul g est partagé entre les couches d'un transformer, face à LoRA, un hypernetwork et ALBERT (Phases 3 et 4) ;
- et, pour écarter l'objection « les cibles n'avaient pas de structure jacobienne », apprentissage à partir d'un professeur dont les matrices sont réellement des tranches d'un champ jacobien (Phase 5).

## Résultats

| Phase | Résultat |
|:---|:---|
| 1. Un point | Reparamétrisation exacte (gap conforme à la théorie), mais équivalente à une LoRA de rang 1 et moins robuste à l'optimisation |
| 2. Un g, m points | Gain modeste (×1,1 à ×1,9 face à W = I), significatif face à la même base sans intégrabilité dans 2 configurations sur 5, uniquement si la cible est intégrable et appartient à la famille gaussienne du générateur. Hors famille, aucun avantage. Sur un champ non intégrable, jusqu'à ×6,4 de perte |
| 3. Transformer 28 k | Pas de différence significative entre méthodes à budget égal ; équivalence non démontrée ; moins stable que LoRA |
| 4. d = 64, L = 6, Shakespeare | La tâche discrimine (dense ≫ ALBERT). Le générateur est à 0,008 nat de LoRA (ns), devant ALBERT de 0,010 (p = 0,061) et indiscernable de l'ablation sans intégrabilité (p = 0,84) |
| 5. Professeur jacobien | Le générateur gagne quand le professeur est dans sa famille (contrôle positif), pas en dehors, où une interpolation linéaire des matrices voisines fait mieux. En distillation, aucun élève n'exploite la structure du professeur |

Conclusion : la contrainte d'intégrabilité n'a pas montré d'utilité pratique. Le générateur fonctionne et tient la comparaison avec LoRA et l'hypernetwork, mais rien n'indique qu'il fasse mieux, ni que le caractère jacobien y soit pour quelque chose : l'ablation (même base gaussienne, sans la contrainte) obtient les mêmes résultats. Le seul avantage mesuré, sur des cibles intégrables de sa propre famille, est presque tautologique. La Phase 5 exclut l'hypothèse d'un manque de structure dans les cibles. Le projet est arrêté.

## Organisation du code

| Fichier | Contenu |
|:---|:---|
| `src/genjac.py`, `src/hypernet.py` | Générateurs jacobiens gaussiens (gradients analytiques), hypernetwork à sortie libre, Adam |
| `src/phase1.py` | Phase 1 : calibrage à un point (E1a, E1b, hessienne) |
| `src/phase2.py` | Phase 2 : 4 familles de cibles × 7 prédicteurs × 8 graines, Wilcoxon apparié, ablation, baselines triviales |
| `src/phase3_model.py`, `phase3_methods.py`, `phase3_train.py` | Transformer NumPy à rétropropagation manuelle ; fournisseurs de W₁ : `gen`, `gen_naive`, `genfree`, `lora`, `hyp`, `albert`, `dense` |
| `src/phase3_analyze.py` | Analyse Phase 3 : appariement par graine, TOST, tests F |
| `scale/jwg_scale.py` | Phase 4 en PyTorch (`selftest`, `train`, `bench`) |
| `src/tables.py` | Génère les tableaux du README à partir des JSON bruts |
| `src/figure_3d.py` | Figure d'en-tête |
| `phase5_teacher_student/` | Phase 5 : professeur jacobien, élèves, prédiction du champ (E1) et distillation (E2) |
| `tests/` | 6 + 5 tests : gradients contre différences finies (générateur, ablation, 7 fournisseurs, transformer), intégrabilité (curl 10⁻¹¹ pour le générateur, 6·10⁻² pour l'ablation), gaussiennes actives à l'init en d = 32 et 128, budgets |

Méthodes comparées :
- `gen` : le générateur, avec une init où les gaussiennes sont actives (géométrie fixée en norme, indépendante de d) ;
- `gen_naive` : centres tirés en N(0, 0,8²) par coordonnée, ce qui rend les gaussiennes inactives en grande dimension (φ_max ≤ 10⁻⁶ en d = 128) ;
- `genfree` : ablation, même base gaussienne mais amplitudes de rang 1 libres, sans intégrabilité ;
- `hyp` : hypernetwork à sortie libre ;
- `albert` : partage total entre couches ;
- `dense` : référence haute.

### Reproduire

```bash
pip install -r requirements.txt           # numpy, matplotlib (+ torch pour la Phase 4)
python3 src/phase1.py ; python3 src/figures.py
python3 src/phase2.py --d 8 --m 4        # idem pour les configs (2,2) (2,4) (2,8) (4,4) (8,4)
python3 src/phase3_train.py --task a --method gen --seed 0 --steps 4000   # 7 méthodes × 5 graines
python3 src/phase3_analyze.py
python3 scale/jwg_scale.py bench --methods gen genfree lora hyp albert dense gen_naive --seeds 0 1 2 3 4 \
    --d 64 --d-ff 256 --L 6 --heads 4 --T 64 --B 16 --steps 3000 --warmup 150 --w1-budget 31744 \
    --corpus shakespeare.txt --device cpu    # tiny shakespeare (~1,1 Mo), non versionné, à placer à la racine
python3 src/tables.py
```

## Phase 1 : un seul point

Une gaussienne, u₀ = (0,5 ; 0,5), a fixé à (0,6 ; 0,8). Variantes : V1 = (x, y, log R), V2 = (x, y) avec R = 0,5 fixé, V3 = V1 + a libre. Baselines : LoRA de rang 1 à a fixé, matrice 2×2 libre.

**E1a. Cibles atteignables (12 instances × 5 graines).** Quand l'optimisation converge, la précision est celle de la machine (pertes finales de l'ordre de 5–7·10⁻¹⁴). Le taux de convergence à 10⁻¹⁰ est de 88 % (V1), 87 % (V2) et 93 % (V3), contre 100 % pour LoRA. Les runs bloqués restent entre 1,8·10⁻¹ et 2,3·10⁻¹ : les centres se sont éloignés de u₀ et les gaussiennes ne se réactivent pas en 4 000 pas. Avec un bruit σ = 0,01 sur la cible, les planchers sont de 9,8·10⁻⁵ pour V1 comme pour LoRA, 3,7·10⁻⁵ pour V3 (a libre absorbe mieux le bruit) et 0 pour la matrice libre.

**E1b. Cibles 2×2 quelconques**, gap comparé à la valeur théorique ‖(I − ââᵀ)(W* − I)‖_F :

| Méthode | gap médian | excès médian | excès max |
|:---|:---:|:---:|:---:|
| V1 (x, y, log R, a fixé) | 1,2694 | 0,0000 | 1,1·10⁻¹⁶ |
| V2 (R fixé) | 1,2694 | ~0 | 0,0223 |
| V3 (+ a libre) | 0,5575 | −0,53 | −0,02 |
| LoRA rang 1 (a fixé) | 1,2694 | 0,0000 | 2,2·10⁻¹⁶ |
| Matrice libre | 0 | — | — |

V1 et LoRA atteignent exactement le gap théorique : sur une cible quelconque, la reparamétrisation ne perd rien et ne gagne rien. V2 est limitée par son ensemble borné (‖v‖ ≤ e^(−1/2)/R₀). V3 passe sous la borne parce que a libre élargit la famille.

**Hessienne de V1 au point convergé (5 instances).** Plus petite valeur propre entre 1,2 et 7,5·10⁻⁷, contre λ₂, λ₃ ∈ [0,24 ; 4,4], soit λ₁/λ₃ ≥ 2,7·10⁻⁸. Il existe donc une direction plate, correspondant aux couples (d, R) qui laissent v inchangé. Voir [`figures/phase1_direction_plate.svg`](figures/phase1_direction_plate.svg).

## Phase 2 : un même g évalué en m points

Budgets appariés (générateur à K(2d+1) paramètres contre hypernetwork) : 20 vs 18 pour d = 2, 45 vs 37 pour d = 4, 153 vs 137 pour d = 8.

Quatre familles de cibles :
- (a) intégrables, dans la famille gaussienne du générateur ;
- (mis) intégrables, hors famille (g* = u + ½B·tanh(Cu+e)) ;
- (b1) matrices indépendantes ;
- (b2) champ lisse non intégrable.

Métrique : RMSE de généralisation sur 100 points non vus, médiane sur 8 graines. Les p sont des tests de Wilcoxon appariés (avec 8 graines, le plus petit p possible est 0,0078), sans correction pour comparaisons multiples. Baselines : W = I, moyenne des matrices d'entraînement, hypernetwork résiduel (W = I + h(u), même prior), hypernetwork H = 8 (hors budget) et même base gaussienne sans intégrabilité. Le curl mesure l'écart à l'intégrabilité ; il est nul pour un champ exactement intégrable.

**(a) Cibles intégrables, dans la famille gaussienne**

| Config | générateur | hypernet | hypernet résiduel | hypernet H=8 | même base sans intégrabilité | W = I | moyenne train | victoires gen/hyp (p) | gen/hyp résiduel (p) | gen/sans-intégrabilité (p) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| d=2, m=2 | 0.287 | 0.580 | 0.539 | 0.396 | 0.404 | 0.306 | 0.332 | 7/8 (p=0.039)** | 7/8 (p=0.023)** | 7/8 (p=0.078) |
| d=2, m=4 | 0.168 | 0.483 | 0.202 | 0.191 | 0.337 | 0.225 | 0.232 | 7/8 (p=0.078) | 6/8 (p=0.195) | 6/8 (p=0.039)** |
| d=2, m=8 | 0.118 | 0.187 | 0.207 | 0.164 | 0.222 | 0.225 | 0.221 | 5/8 (p=0.461) | 7/8 (p=0.055) | 7/8 (p=0.055) |
| d=4, m=4 | 0.068 | 0.292 | 0.268 | 0.311 | 0.189 | 0.113 | 0.117 | 7/8 (p=0.109) | 7/8 (p=0.055) | 7/8 (p=0.148) |
| d=8, m=4 | 0.015 | 0.253 | 0.282 | 0.242 | 0.039 | 0.027 | 0.029 | 8/8 (p=0.008)** | 8/8 (p=0.008)** | 7/8 (p=0.016)** |

Observations :
1. L'hypernetwork à sortie libre fait pire que W = I dans les 5 configurations. Le comparer seul au générateur surestime les gains : le ×17 obtenu à d = 8 devient ×1,8 face à W = I.
2. Le générateur bat W = I et la moyenne partout, mais seulement d'un facteur 1,07 à 1,9.
3. Face à la même base sans intégrabilité, il est meilleur en médiane dans les 5 configurations, significativement dans 2 (d = 2, m = 4 et d = 8).
4. Face à l'hypernetwork résiduel, qui partage le même prior, le gain n'est significatif qu'à d = 2, m = 2 et à d = 8.

**(mis) Cibles intégrables, hors famille gaussienne**

| Config | générateur | hypernet | hypernet résiduel | hypernet H=8 | même base sans intégrabilité | W = I | moyenne train | victoires gen/hyp (p) | gen/hyp résiduel (p) | gen/sans-intégrabilité (p) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| d=2, m=2 | 0.334 | 0.567 | 0.320 | 0.343 | 0.231 | 0.309 | 0.225 | 6/8 (p=0.312) | 5/8 (p=0.383) | 2/8 (p=0.148) |
| d=2, m=4 | 0.298 | 0.315 | 0.323 | 0.329 | 0.393 | 0.253 | 0.208 | 4/8 (p=0.641) | 6/8 (p=0.148) | 4/8 (p=0.312) |
| d=2, m=8 | 0.278 | 0.141 | 0.114 | 0.120 | 0.225 | 0.211 | 0.134 | 3/8 (p=0.383) | 2/8 (p=0.109) | 4/8 (p=0.945) |
| d=4, m=4 | 0.287 | 0.334 | 0.298 | 0.371 | 0.233 | 0.216 | 0.141 | 6/8 (p=0.383) | 4/8 (p=0.844) | 4/8 (p=0.641) |
| d=8, m=4 | 0.137 | 0.243 | 0.214 | 0.304 | 0.156 | 0.141 | 0.095 | 8/8 (p=0.008)** | 7/8 (p=0.039)** | 3/8 (p=0.641) |

Quand la cible est intégrable sans être gaussienne, l'avantage disparaît. La moyenne des matrices d'entraînement bat le générateur dans les 5 configurations, et à d = 2, m = 8 l'hypernetwork résiduel fait 2,4 fois mieux. L'intégrabilité ne suffit pas ; il faut aussi que la famille de g corresponde à la cible.

**(b2) Champ lisse non intégrable**

| Config | générateur | hypernet | hypernet résiduel | hypernet H=8 | même base sans intégrabilité | W = I | moyenne train | victoires gen/hyp (p) | gen/hyp résiduel (p) | gen/sans-intégrabilité (p) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| d=2, m=2 | 0.314 | 0.353 | 0.368 | 0.275 | 0.209 | 0.245 | 0.226 | 5/8 (p=0.945) | 4/8 (p=0.844) | 1/8 (p=0.055) |
| d=2, m=4 | 0.488 | 0.076 | 0.063 | 0.094 | 0.315 | 0.270 | 0.236 | 0/8 (p=0.008)** | 0/8 (p=0.008)** | 2/8 (p=0.148) |
| d=2, m=8 | 0.200 | 0.031 | 0.026 | 0.029 | 0.090 | 0.302 | 0.195 | 0/8 (p=0.008)** | 0/8 (p=0.008)** | 3/8 (p=0.109) |
| d=4, m=4 | 0.455 | 0.391 | 0.368 | 0.414 | 0.468 | 0.333 | 0.333 | 1/8 (p=0.148) | 2/8 (p=0.109) | 4/8 (p=0.945) |
| d=8, m=4 | 0.431 | 0.455 | 0.464 | 0.627 | 0.420 | 0.346 | 0.359 | 6/8 (p=0.195) | 6/8 (p=0.078) | 4/8 (p=1.000) |

Sur ce type de cible, la contrainte coûte cher : jusqu'à 6,4 fois pire que l'hypernetwork à d = 2 (p = 0,008). Sur des matrices indépendantes (b1), elle ne coûte presque rien dès m ≥ 4.

**Curl des champs appris (famille a) et dispersion entre graines**

| Config | curl générateur | curl hypernet | σ générateur | σ hypernet |
|:---|:---:|:---:|:---:|:---:|
| d=2, m=2 | 6.3e-11 | 0.44 | 0.116 | 0.152 |
| d=2, m=4 | 2.3e-11 | 0.54 | 0.252 | 0.301 |
| d=2, m=8 | 1.7e-11 | 0.16 | 0.114 | 0.084 |
| d=4, m=4 | 1.1e-11 | 0.19 | 0.131 | 0.119 |
| d=8, m=4 | 4.2e-12 | 0.13 | 0.007 | 0.113 |

Le curl du générateur est de l'ordre de 10⁻¹¹, ce qui est vrai par construction et sert seulement de vérification. Figure : [`figures/phase2_controles.svg`](figures/phase2_controles.svg).

## Phase 3 : transformer complet

Transformer de type GPT au niveau caractère, en NumPy (d = 32, d_ff = 64, L = 4, environ 28 k paramètres). La matrice W₁ de chaque FFN est produite par chacune des méthodes, à budget apparié (±2,3 %). Tâche A (texte synthétique zipfien), 4 000 pas, 5 graines, même init et mêmes lots pour toutes les méthodes.

| Méthode | params W₁ | perte de validation (moy. ± σ) | n graines | dist. inter-couches | Δ(gen − méthode) [IC 90 %] | p apparié |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense (référence haute) | 8,192 | 1.0003 ± 0.0079 | 5 | 156.3 % | +0.0084 [+0.0046 ; +0.0122] | 0.009 |
| LoRA ×L | 2,816 | 1.0123 ± 0.0012 | 5 | 31.3 % | -0.0036 [-0.0085 ; +0.0012] | 0.182 |
| Hypernetwork | 2,806 | 1.0082 ± 0.0055 | 5 | 24.8 % | +0.0005 [-0.0075 ; +0.0085] | 0.905 |
| Générateur jacobien (init active) | 2,758 | 1.0087 ± 0.0045 | 5 | 22.0 % | — | — |
| Ablation : même base, sans intégrabilité | 2,821 | 1.0099 ± 0.0035 | 5 | 18.3 % | -0.0012 [-0.0058 ; +0.0035] | 0.617 |
| ALBERT (partage total) | 2,048 | 1.0136 ± 0.0040 | 5 | 0.0 % | -0.0049 [-0.0107 ; +0.0009] | 0.146 |
| Générateur, init naïve (gaussiennes inactives) | 2,758 | 1.0141 ± 0.0042 | 5 | 6.5 % | -0.0054 [-0.0124 ; +0.0016] | 0.177 |

Discrimination : Δ(dense − ALBERT) = −0,0133 (p = 0,033).

Remarques :
1. Les six méthodes compressées sont toutes dans un intervalle de 0,006 nat et aucune différence n'est significative. L'équivalence n'est démontrée pour aucune (tous les IC 90 % dépassent ±0,005).
2. La tâche discrimine tout juste (dense devant ALBERT, p = 0,033).
3. LoRA est plus stable : σ = 0,0012 contre 0,0045 pour le générateur (test F bilatéral, p = 0,03, non corrigé). Pas de différence de stabilité avec l'hypernetwork (p = 0,69).
4. L'ablation sans intégrabilité donne le même résultat (p = 0,62).
5. L'init naïve fait un peu moins bien (1,0141 contre 1,0087, Δ = −0,005, p = 0,18), dans le sens attendu mais sans significativité à n = 5.
6. Avec l'init active, les couches diffèrent de 22 % (6,5 % avec l'init naïve, 31 % pour LoRA). Le générateur ne se réduit donc pas à un partage total de type ALBERT.

Figure : [`figures/phase3_courbes.svg`](figures/phase3_courbes.svg).

## Phase 4 (version réduite) : d = 64, L = 6, Shakespeare

Le test prévu (d = 128, sur GPU) n'a pas pu être lancé faute de matériel (CPU, 2 cœurs). Version réduite : d = 64, d_ff = 256, L = 6, T = 64, B = 16, 3 000 pas, corpus tiny Shakespeare (1,1 M caractères, V = 65), 7 méthodes × 5 graines, budget W₁ de 31 744 paramètres (98 304 pour le dense).

Critères fixés avant les runs : discrimination (dense meilleur qu'ALBERT), équivalence TOST à ±0,005, supériorité sur ALBERT, supériorité sur l'ablation, différenciation des couches supérieure à 5 %.

| Méthode | params W₁ | perte de validation (moy. ± σ) | n graines | dist. inter-couches | Δ(gen − méthode) [IC 90 %] | p apparié |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense (référence haute) | 98,304 | 1.7292 ± 0.0112 | 5 | 139.7 % | +0.0339 [+0.0214 ; +0.0465] | 0.005 |
| LoRA ×L | 31,744 | 1.7554 ± 0.0129 | 5 | 48.1 % | +0.0078 [-0.0018 ; +0.0174] | 0.159 |
| Hypernetwork | 31,628 | 1.7684 ± 0.0076 | 5 | 20.8 % | -0.0052 [-0.0171 ; +0.0067] | 0.403 |
| Générateur jacobien (init active) | 31,855 | 1.7632 ± 0.0152 | 5 | 38.5 % | — | — |
| Ablation : même base, sans intégrabilité | 31,783 | 1.7617 ± 0.0138 | 5 | 27.7 % | +0.0015 [-0.0135 ; +0.0165] | 0.842 |
| ALBERT (partage total) | 16,384 | 1.7732 ± 0.0105 | 5 | 0.0 % | -0.0100 [-0.0183 ; -0.0017] | 0.061 |
| Générateur, init naïve (gaussiennes inactives) | 31,855 | 1.7676 ± 0.0084 | 5 | 5.1 % | -0.0044 [-0.0180 ; +0.0092] | 0.525 |

Résultats par critère :
- Discrimination : Δ(dense − ALBERT) = −0,0440 (p < 0,001), la tâche discrimine.
- Équivalence avec LoRA (Δ = +0,0078, IC 90 % [−0,0018 ; +0,0174]) et avec l'hypernetwork (Δ = −0,0052, IC 90 % [−0,0171 ; +0,0067]) : non démontrée, intervalles trop larges.
- Comparaison avec ALBERT : Δ = −0,0100 (p = 0,061). Tendance seulement, et ALBERT a deux fois moins de paramètres.
- Comparaison avec l'ablation : Δ = +0,0015 (p = 0,842). La contrainte d'intégrabilité ne fait pas de différence.
- Différenciation des couches : 38,5 % (5 % avec l'init naïve), φ_max final moyen de 0,96. Cela ne se traduit pas sur la perte (init naïve contre active : Δ = −0,004, p = 0,53).

Compresser W₁ (×3, ou ×6 pour ALBERT) coûte entre 0,026 nat (LoRA) et 0,044 nat (ALBERT) par rapport au dense ; le générateur se situe à 0,034. Figures : [`scale/results_scale/courbes_shakespeare.svg`](scale/results_scale/courbes_shakespeare.svg), [`figures/phase4_reduite_courbes.svg`](figures/phase4_reduite_courbes.svg).

## Phase 5 : professeur jacobien

Les phases précédentes laissaient une explication possible au résultat négatif : les cibles testées n'avaient peut-être pas de vraie structure jacobienne. La Phase 5 construit donc un professeur dont les W₁ sont des tranches d'un champ de jacobiennes W*(u) = J_h(u), et des élèves qui apprennent de lui (LoRA, générateur, ablation, hypernetwork, ALBERT), avec le même code et les mêmes budgets qu'en Phase 3. Code et résultats dans [`phase5_teacher_student/`](phase5_teacher_student/).

Deux professeurs :
- gauss : h = M·u + Σ aₖφₖ(u). Le champ appartient exactement à la famille du générateur. C'est le contrôle positif : si le générateur ne gagne pas ici, le harnais est défectueux.
- mlp : h = M·u + s·E·tanhMLP(u). Intégrable par construction (curl 4,6·10⁻⁷) mais hors de la famille gaussienne. C'est le test proprement dit.

Le reste du professeur (attention, W₂, LayerNorm, embeddings) est entraîné sur la tâche A avec W₁ figée aux tranches jacobiennes, ce qui en fait un modèle fonctionnel (CE ≈ 1,51 nat). Sept auto-tests vérifient les gradients de la distillation et des ajusteurs contre des différences finies, l'intégrabilité du professeur MLP et le fait que ses couches sont distinctes.

### E1 : prédire le champ de poids

On observe m = 6 points (uᵢ, W*(uᵢ)), on en masque un, et chaque méthode doit prédire la matrice en ce point. LoRA ne peut pas faire ce test par construction. Deux baselines simples s'ajoutent : interpolation linéaire des matrices voisines (lerp) et moyenne. 5 graines, 3 redémarrages, Adam sur 3 000 pas. Erreur relative ‖pred − W*‖/‖W*‖, médiane :

| Professeur | Point masqué | gen | sans intégrabilité | hypernet | lerp | moyenne |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gauss (dans la famille) | 2 (interpolation) | **0,020** | 0,168 | 0,554 | 0,310 | 0,501 |
| gauss | 5 (extrapolation) | **0,019** | 0,319 | 0,672 | 0,228 | 0,427 |
| mlp (hors famille) | 2 (interpolation) | 0,016 | 0,015 | 0,029 | **0,006** | 0,022 |
| mlp | 5 (extrapolation) | 0,044 | 0,044 | 0,054 | **0,028** | 0,074 |

Sur le professeur gauss, le générateur gagne dans les 5 graines contre chaque concurrent, avec une erreur 9 à 17 fois plus faible que l'ablation. Le harnais fonctionne, et cela confirme que le gain observé en Phase 2 venait de la correspondance de famille et non de l'intégrabilité.

Sur le professeur mlp, le générateur fait jeu égal avec l'ablation (Δ = +0,0015 en interpolation, 0/5 graines ; Δ = −0,0001 en extrapolation, 3/5) et il est battu par lerp dans toutes les graines, d'un facteur 1,6 à 2,9. Un champ lisse et presque linéaire se prédit mieux en interpolant les matrices voisines qu'avec des bosses gaussiennes ; l'intégrabilité n'apporte rien de plus que la régularité. Le générateur bat l'hypernetwork (5/5 et 4/5), mais lerp ne demande aucun entraînement.

### E2 : distillation

Les élèves imitent le professeur par KL sur les lots de la tâche A (transformer NumPy de 28 k paramètres, T = 32, 1 000 pas, lots et init appariés par graine, 3 graines). Médiane sur 3 graines, CE du professeur ≈ 1,51 nat :

| Professeur | Méthode | KL@100 | KL@300 | KL final | ‖W₁ − W₁*‖ |
|:---|:---|:---:|:---:|:---:|:---:|
| mlp | gen | 0,472 | 0,226 | 0,074 | 147 % |
| mlp | sans intégrabilité | 0,453 | 0,189 | 0,073 | 145 % |
| mlp | LoRA | 0,453 | 0,219 | 0,070 | 146 % |
| mlp | hypernet | 0,452 | 0,213 | 0,072 | 146 % |
| mlp | ALBERT | 0,455 | 0,222 | **0,069** | 145 % |
| gauss | gen | 0,480 | 0,195 | 0,066 | 142 % |
| gauss | sans intégrabilité | 0,455 | 0,196 | 0,067 | 141 % |
| gauss | LoRA | 0,471 | 0,198 | **0,065** | 141 % |
| gauss | hypernet | 0,466 | 0,192 | 0,067 | 141 % |
| gauss | ALBERT | 0,469 | 0,193 | 0,068 | 140 % |

Toutes les méthodes finissent à environ 0,002 nat de KL les unes des autres, y compris quand le professeur appartient exactement à la famille du générateur (gauss : 0,0658 pour le générateur contre 0,0653 pour LoRA). Aucune différence n'est cohérente sur les 3 graines ; le générateur ne bat LoRA dans les 3 graines pour aucune configuration.

La distillation par gradients de tâche ne transmet pas la structure du champ : les élèves trouvent des matrices fonctionnellement équivalentes mais éloignées de 140 % des W₁ du professeur. Sur mlp, le générateur démarre plus lentement (KL@300 de 0,226 contre 0,189 pour l'ablation, 2 graines sur 3).

### Bilan

L'hypothèse d'un manque de structure dans les cibles ne tient pas. Même avec un professeur qui est réellement un champ de jacobiennes, le générateur ne gagne en prédiction du champ que si la famille correspond, et perd face à une interpolation linéaire sinon. En distillation, il ne fait jamais mieux que LoRA, même dans le cas le plus favorable. Pour qu'un prior jacobien soit utile, il faudrait un problème de régression de champ, interrogé hors des points d'entraînement, avec la bonne famille de fonctions, et le résultat devrait encore battre lerp.

### Limites de la Phase 5

3 graines et 1 000 pas en E2, un seul corpus, pas de correction pour comparaisons multiples ; les différences en E2 sont de l'ordre du bruit entre graines. Les élèves ne voient jamais les W₁ du professeur (gradients de tâche uniquement) : une distillation combinée à une régression sur les poids n'a pas été testée. En E1, seulement 6 points, d = 32, et un champ MLP presque linéaire sur le segment observé. Un champ plus courbé pourrait désavantager lerp, mais l'égalité entre générateur et ablation ne dépend pas de la courbure.

Reproduire (depuis `phase5_teacher_student/`) : `python src/selftest.py` (7/7), `python src/e1_field.py` (environ 3 min), `python src/e2_distill.py --parent mlp --method gen --seed 0` (environ 35 s par run, 2 professeurs × 5 méthodes × 3 graines), puis `python src/analyze.py`.

## Limites générales

- Échelle : transformers de 28 k puis environ 1 M de paramètres ; seule W₁ est contrainte (ni W₂ ni l'attention) ; un seul type de générateur (gaussiennes analytiques). Le générateur MLP (qui nécessite des dérivées secondes) et le cas g scalaire (W = hessienne symétrique) n'ont pas été testés.
- Statistiques : 5 graines, 3 000 à 4 000 pas, un corpus synthétique et un corpus réel, pas de correction pour comparaisons multiples, intervalles d'équivalence trop larges pour conclure à l'égalité.
- Phase 2 : interpolation uniquement, cibles de faible dimension (d ≤ 8).
- Phase 4 : version réduite (d = 64 au lieu de 128, 3 000 pas, CPU). La commande ci-dessus avec `--d 128 --d-ff 512 --L 6 --steps 6000` est prête pour un run sur GPU.
- Phase 5 : un seul professeur MLP, 3 graines en distillation (voir plus haut).
