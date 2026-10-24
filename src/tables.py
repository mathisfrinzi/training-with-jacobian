# -*- coding: utf-8 -*-
"""Génère (depuis les JSON bruts) les tableaux Markdown de la Phase 2 publiés dans le README.
Usage : python3 src/tables.py   → imprime les tableaux ; aucune valeur n'est saisie à la main."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "results_phase2.json").read_text())


def fmt(x):
    return f"{x:.3f}" if x >= 0.01 else f"{x:.4f}"


def star(p):
    return "" if p is None else ("**" if p < 0.05 else "")


for fam, title in [("a", "Famille (a) — cibles intégrables DANS la famille gaussienne du générateur"),
                   ("mis", "Famille (mis) — cibles intégrables HORS famille gaussienne (g* = u + ½B·tanh(Cu+e))"),
                   ("b2", "Famille (b2) — champ lisse NON intégrable")]:
    print(f"\n### {title}\nRMSE de généralisation (100 points non vus), médiane sur 8 graines.\n")
    print("| Config | générateur | hypernet | hypernet résiduel | hypernet H=8 | même base sans intégrabilité | **W = I** | moyenne train | victoires gen/hyp (p) | gen/hyp résiduel (p) | gen/sans-intégrabilité (p) |")
    print("|:---|" + ":---:|" * 10)
    for c in d["configs"]:
        f = c["families"].get(fam)
        if not f or "controls" not in f:
            continue
        m = {k: v["med"] for k, v in f["controls"].items()}
        t = f["tests"]
        h, g, hr = t["gen_vs_hyp"], t["gen_vs_gaussfree"], t["gen_vs_hyp_res"]
        print(f"| d={c['d']}, m={c['m']} | {fmt(m['gen'])} | {fmt(m['hyp'])} | {fmt(m['hyp_res'])} | {fmt(m['hyp_big'])} | "
              f"{fmt(m['gaussfree'])} | {fmt(m['identity'])} | {fmt(m['const_mean'])} | "
              f"{h['gen_wins']}/8 (p={h['wilcoxon_p']:.3f}){star(h['wilcoxon_p'])} | {hr['gen_wins']}/8 (p={hr['wilcoxon_p']:.3f}){star(hr['wilcoxon_p'])} | {g['gen_wins']}/8 (p={g['wilcoxon_p']:.3f}){star(g['wilcoxon_p'])} |")

print("\n### Intégrabilité (curl, famille a) et stabilité\n")
print("| Config | curl générateur | curl hypernet | σ générateur | σ hypernet |")
print("|:---|:---:|:---:|:---:|:---:|")
for c in d["configs"]:
    f = c["families"]["a"]
    print(f"| d={c['d']}, m={c['m']} | {f['generator']['curl_med']:.1e} | {f['hypernet']['curl_med']:.2f} | "
          f"{f['generator']['gen_std']:.3f} | {f['hypernet']['gen_std']:.3f} |")


# ---------------------------------------------------------------------------- Phase 3 et Phase 4
LAB = {"dense": "Dense (référence haute)", "lora": "LoRA ×L", "hyp": "Hypernetwork", "gen": "**Générateur jacobien (init vivante)**",
       "genfree": "Ablation : même base, sans intégrabilité", "albert": "ALBERT (partage total)",
       "gen_naive": "Générateur, init naïve (gaussiennes mortes)"}
ORDER = ["dense", "lora", "hyp", "gen", "genfree", "albert", "gen_naive"]


def phase_table(summary_path, key_pairs, title, pairs_key="paired_vs_gen"):
    p = Path(summary_path)
    if not p.exists():
        return
    s = json.loads(p.read_text())
    print(f"\n### {title}\n")
    if "table" in s:                                    # format Phase 4 (jwg_scale)
        tab = {m: dict(val_mean=r["val_mean"], val_std=r["val_std"], n=r["n"], ff1=r["ff1_params"],
                       dist=r["dist_intercouches"]) for m, r in s["table"].items()}
        pairs = {k.replace("gen_vs_", ""): dict(delta=v["delta_moyen"], p=v["p"], ci=v["tost"]["ci90"])
                 for k, v in s["paired"].items()}
    else:                                               # format Phase 3 (phase3_analyze)
        tab = {m: dict(val_mean=r["val_mean"], val_std=r["val_std"], n=len(r["vals"]), ff1=r["ff1_params"],
                       dist=r["mean_interlayer_dist"]) for m, r in s["methods"].items()}
        pairs = {k: dict(delta=v["delta"], p=v["p"], ci=v["ci90"]) for k, v in s["paired_vs_gen"].items()}
    print("| Méthode | params W₁ | perte de validation (moy. ± σ) | n graines | dist. inter-couches | Δ(gen − méthode) [IC 90 %] | p apparié |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in ORDER:
        if m not in tab:
            continue
        r = tab[m]
        pr = pairs.get(m)
        d_ = f"{pr['delta']:+.4f} [{pr['ci'][0]:+.4f} ; {pr['ci'][1]:+.4f}]" if pr else "—"
        pv = f"{pr['p']:.3f}" if pr else "—"
        print(f"| {LAB[m]} | {r['ff1']:,} | {r['val_mean']:.4f} ± {r['val_std']:.4f} | {r['n']} | {r['dist']*100:.1f} % | {d_} | {pv} |")
    if "discrimine" in s:
        dv = s["paired_other"]["dense_vs_albert"]
        print(f"\nDiscrimination : Δ(dense − ALBERT) = {dv['delta']:+.4f} (p = {dv['p']:.3f}) → "
              f"{'tâche discriminante' if s['discrimine'] else 'tâche NON discriminante'}.")
    if "verdict" in s:
        print()
        for l in s["verdict"]["lines"]:
            print(f"- {l}")


phase_table(ROOT / "results" / "phase3" / "summary.json", None, "Phase 3 — transformer 28 k, tâche A, 4 000 pas")
phase_table(ROOT / "scale" / "results_scale" / "summary_shakespeare.json", None,
            "Phase 4 réduite — transformer d=64, L=6, Shakespeare, 3 000 pas")
