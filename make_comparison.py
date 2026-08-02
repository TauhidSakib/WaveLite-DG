# Genuine baseline comparison figures from the real saved results.
import json, glob, os
import numpy as np
from figlib import *   # style + palette + save + torch
import torch

OUTB = "outputs/baselines"
# model display order; ckpt for exact param count
MODELS = [
    ("WaveLite-DG", "ours",     "outputs/wavemamba_dg_best.pt"),
    ("U-Net",        "unet",     f"{OUTB}/unet_best.pt"),
    ("PraNet",       "pranet",   f"{OUTB}/pranet_best.pt"),
    ("Polyp-PVT",    "polyppvt", f"{OUTB}/polyppvt_best.pt"),
    ("SSFormer-S",   "ssformer", f"{OUTB}/ssformer_best.pt"),
    ("VM-UNet",      "vmunet",   f"{OUTB}/vmunet_best.pt"),
]
DSETS = ["Kvasir-SEG","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]
UNSEEN = ["CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]
SHORTD = {"Kvasir-SEG":"Kvasir*","CVC-ClinicDB":"ClinicDB*","CVC-ColonDB":"ColonDB","ETIS-LaribPolypDB":"ETIS","CVC-300":"CVC-300"}

def params_M(ckpt):
    if not os.path.exists(ckpt): return None
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))/1e6

# load available results
rows=[]  # (name, key, dice_dict, params, perimage_dict)
for name, key, ckpt in MODELS:
    rp = f"{OUTB}/{key}_report.json"
    if not os.path.exists(rp): continue
    rep = json.load(open(rp))["report"]
    dd = {d: rep[d][1] for d in DSETS}
    pm = params_M(ckpt)
    pi = json.load(open(f"{OUTB}/{key}_perimage.json")) if os.path.exists(f"{OUTB}/{key}_perimage.json") else None
    rows.append((name, key, dd, pm, pi))
names = [r[0] for r in rows]
print("models:", names)
MCOL = {"WaveLite-DG":BLUE, "U-Net":"#8a8a8a", "PraNet":LGREEN, "Polyp-PVT":ORNG, "SSFormer-S":LBLUE, "VM-UNet":MAG}

# ---------------- FIG A: grouped bars + heatmap ----------------
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9.6, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
axb = fig.add_subplot(gs[0]); axh = fig.add_subplot(gs[1])
x = np.arange(len(DSETS)); nb = len(rows); w = 0.8/nb
axb.axvspan(1.5, len(DSETS)-0.4, color=UNSEEN_L if (UNSEEN_L:="#F6C79A") else ORNG, alpha=0.18, lw=0)
for i,(name,key,dd,pm,pi) in enumerate(rows):
    vals=[dd[d] for d in DSETS]
    off=(i-(nb-1)/2)*w
    hl = name=="WaveLite-DG"
    axb.bar(x+off, vals, w, color=MCOL[name], edgecolor="black" if hl else "white",
            lw=1.0 if hl else 0.4, zorder=3, label=name+(" (ours)" if hl else ""))
axb.set_xticks(x); axb.set_xticklabels([SHORTD[d] for d in DSETS], fontsize=7.5, rotation=18, ha="right")
axb.set_ylabel("Dice"); axb.set_ylim(0.55, 1.0)
axb.axvline(1.5, color=BASE, lw=0.8, ls=(0,(4,3)))
axb.text(3.0, 0.985, "UNSEEN (cross-centre)", ha="center", fontsize=7.5, fontweight="bold", color="#B4611A")
axb.set_title("(a) Dice across datasets  (* = seen in training)")
axb.legend(fontsize=6.6, ncol=2, loc="lower left", handlelength=1.1, columnspacing=0.9)
axb.grid(axis="y", color=GRID, lw=0.5); axb.set_axisbelow(True); full_box(axb)
# heatmap
M = np.array([[dd[d] for d in DSETS] for (_,_,dd,_,_) in rows])
im = axh.imshow(M, cmap="Blues", vmin=0.55, vmax=1.0, aspect="auto")
axh.set_xticks(range(len(DSETS))); axh.set_xticklabels([SHORTD[d] for d in DSETS], fontsize=7, rotation=30, ha="right")
axh.set_yticks(range(nb)); axh.set_yticklabels(names, fontsize=7.5)
for r in range(nb):
    for c in range(len(DSETS)):
        axh.text(c, r, f"{M[r,c]:.3f}", ha="center", va="center", fontsize=6.3,
                 color="white" if M[r,c]>0.85 else INK, fontweight="bold" if names[r]=="WaveLite-DG" else "normal")
axh.set_title("(b) Dice heatmap"); axh.tick_params(length=0)
cb=fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.6)
fig.tight_layout(); save(fig, "fig_comparison_bars")

# ---------------- FIG B: efficiency (mean unseen Dice vs params) ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name,key,dd,pm,pi in rows:
    if pm is None: continue
    mu = float(np.mean([dd[d] for d in UNSEEN]))
    hl = name=="WaveLite-DG"; right = pm > 26
    ax.scatter(pm, mu, s=170 if hl else 90, color=MCOL[name], edgecolor="black", lw=1.2 if hl else 0.6,
               marker="*" if hl else "o", zorder=5)
    ax.annotate(name+(" (ours)" if hl else ""), (pm, mu), textcoords="offset points",
                xytext=(-9 if right else 9, 5), ha="right" if right else "left",
                fontsize=7.5, fontweight="bold" if hl else "normal", color=INK if hl else INK2)
ax.set_xlim(5, 36)
ax.set_xlabel("Parameters (M)  ← smaller is better"); ax.set_ylabel("Mean unseen Dice  → better")
ax.set_title("Efficiency vs. generalization  (top-left is best)")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); full_box(ax)
fig.tight_layout(); save(fig, "fig_efficiency")

# ---------------- FIG C: per-image significance (pooled unseen, Wilcoxon vs ours) ----------------
from scipy.stats import wilcoxon
pooled = {}
for name,key,dd,pm,pi in rows:
    if pi is None: continue
    pooled[name] = np.concatenate([np.array(pi[d]) for d in UNSEEN])
if "WaveLite-DG" in pooled:
    ours = pooled["WaveLite-DG"]
    fig, ax = plt.subplots(1,2, figsize=(9.0,3.6), gridspec_kw=dict(width_ratios=[1.25,1.0]))
    order=[n for n in names if n in pooled]
    data=[pooled[n] for n in order]
    vp=ax[0].violinplot(data, positions=np.arange(len(order)), widths=0.8, showextrema=False)
    for b,n in zip(vp["bodies"],order): b.set_facecolor(MCOL[n]); b.set_alpha(0.5); b.set_edgecolor(MCOL[n])
    bp=ax[0].boxplot(data, positions=np.arange(len(order)), widths=0.15, showfliers=False, patch_artist=True,
                     medianprops=dict(color="white",lw=1.2), boxprops=dict(facecolor="#333",edgecolor="#333",lw=0.5),
                     whiskerprops=dict(color="#333",lw=0.7), capprops=dict(color="#333",lw=0.7))
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([n+("\n(ours)" if n=="WaveLite-DG" else "") for n in order], fontsize=7, rotation=15, ha="right")
    ax[0].set_ylabel("Per-image Dice (pooled unseen)"); ax[0].set_ylim(0,1.03)
    ax[0].set_title("(a) Per-image Dice distribution"); ax[0].grid(axis="y",color=GRID,lw=0.5); ax[0].set_axisbelow(True); full_box(ax[0])
    # Wilcoxon ours vs each baseline
    others=[n for n in order if n!="WaveLite-DG"]; deltas=[]; ps=[]
    for n in others:
        d=float(np.mean(ours)-np.mean(pooled[n]))
        try: _,p=wilcoxon(ours, pooled[n])
        except Exception: p=np.nan
        deltas.append(d); ps.append(p)
    yy=np.arange(len(others))
    cols=[GREEN_D if d>0 else RED_D for d in deltas]
    ax[1].barh(yy, deltas, color=cols, edgecolor="black", lw=0.5, zorder=3)
    for i,(d,p) in enumerate(zip(deltas,ps)):
        star = "n.s." if (np.isnan(p) or p>=0.05) else ("***" if p<1e-3 else ("**" if p<1e-2 else "*"))
        ax[1].text(max(d,0)+0.003, i, f"{d:+.3f}  (p={p:.1e} {star})", va="center", ha="left", fontsize=6.6)
    ax[1].axvline(0,color="#333",lw=0.8); ax[1].set_yticks(yy); ax[1].set_yticklabels(others, fontsize=7.5)
    ax[1].set_xlim(min(deltas)-0.012, 0.11)
    ax[1].set_xlabel("Mean Dice difference (ours − baseline)"); ax[1].set_title("(b) Paired Wilcoxon vs. WaveLite-DG")
    ax[1].grid(axis="x",color=GRID,lw=0.5); ax[1].set_axisbelow(True); full_box(ax[1])
    fig.tight_layout(); save(fig, "fig_significance")

print("\nPARAMS (M):")
for name,key,dd,pm,pi in rows: print(f"  {name:14s} {pm:.2f}" if pm else f"  {name:14s} n/a")
