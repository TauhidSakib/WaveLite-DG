# Stage 3: render the qualitative comparison grid (demo layout, REAL predictions + REAL Dice).
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np
from figlib import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sel   = json.load(open("outputs/qual_selection.json"))
cache = np.load("outputs/qual_cache.npz")
dice  = json.load(open("outputs/qual_dice_native.json"))
inputs, gts = cache["inputs"], cache["gts"]

# column spec: (display, key, is_ours) — VM-UNet added only if its stage-2 output exists
COLS = [("WaveLite-DG\n(Ours)","ours",True), ("PraNet","pranet",False),
        ("Polyp-PVT","polyppvt",False), ("SSFormer-S","ssformer",False)]
vm_npz = "outputs/qual_vmunet.npz"; vm_dice = "outputs/qual_dice_vmunet.json"
preds = {k: cache[f"pred_{k}"] for _, k, _ in COLS}
if os.path.exists(vm_npz) and os.path.exists(vm_dice):
    preds["vmunet"] = np.load(vm_npz)["pred_vmunet"]; dice["vmunet"] = json.load(open(vm_dice))
    COLS.append(("VM-UNet","vmunet",False))

RED = "#C0392B"; nrow = len(sel); ncol = 2 + len(COLS)
fig, ax = plt.subplots(nrow, ncol, figsize=(2.05*ncol, 2.28*nrow))
heads = ["Input","Ground Truth"] + [c[0] for c in COLS]

for r, s in enumerate(sel):
    row_best = max(dice[k][r] for _, k, _ in COLS)   # honest: bold the TRUE best per row
    # input
    ax[r,0].imshow(inputs[r]);
    ax[r,0].set_ylabel(f"{s['dataset']}\n({s['kind']})", fontsize=9,
                       color=(SEEN if s['kind']=='seen' else '#B4611A'), fontweight="bold", labelpad=6)
    # GT
    ax[r,1].imshow(gts[r], cmap="gray", vmin=0, vmax=1)
    # model preds
    for c,(disp,key,ours) in enumerate(COLS):
        a = ax[r, 2+c]; a.imshow(preds[key][r], cmap="gray", vmin=0, vmax=1)
        d = dice[key][r]; best = abs(d-row_best) < 1e-9
        col = RED if ours else INK
        a.text(0.5, -0.085, f"Dice {d:.3f}", transform=a.transAxes, ha="center", va="top",
               fontsize=9, color=col, fontweight="bold" if (ours or best) else "normal")
        if ours:  # identity border on the proposed method's column (not a claim of winning)
            for sp in a.spines.values(): sp.set_color(RED); sp.set_linewidth(2.2); sp.set_visible(True)
    for c in range(ncol):
        a = ax[r,c]
        if not (c>=2 and COLS[c-2][2]):
            for sp in a.spines.values(): sp.set_color("#cccccc"); sp.set_linewidth(0.6)
        a.set_xticks([]); a.set_yticks([])
        if r==0:
            hc = RED if heads[c].startswith("WaveLite") else "black"
            a.set_title(heads[c], fontsize=10.5, fontweight="bold", color=hc, pad=6)

fig.suptitle("Qualitative comparison on polyp segmentation (real predictions)", fontsize=13, y=0.998)
fig.tight_layout(rect=[0,0.01,1,0.985]); fig.subplots_adjust(hspace=0.16, wspace=0.05)
os.makedirs("outputs/figures_final", exist_ok=True)
fig.savefig("outputs/figures_final/Figure_qualitative_comparison.pdf", bbox_inches="tight")
fig.savefig("outputs/figures_final/Figure_qualitative_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("SAVED Figure_qualitative_comparison  (models:", [c[1] for c in COLS], ")")
# honest per-row winner summary
for r,s in enumerate(sel):
    ranked = sorted(((dice[k][r],d) for d,k,_ in COLS), reverse=True)
    print(f"  {s['dataset']:20s} best={ranked[0][1]} ({ranked[0][0]:.3f})  ours={dice['ours'][r]:.3f}")
