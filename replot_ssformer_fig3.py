# Re-render SSFormer-S Fig 3 from saved history (no retrain). Fixes annotation overlap with y-axis.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np
from figlib import *
import matplotlib.pyplot as plt

hist = json.load(open("outputs/baselines/ssformer_history.json"))
TAG = "SSFormer-S (baseline)"
e = np.arange(1, len(hist["train_loss"]) + 1)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

# (a) loss convergence
ax[0].plot(e, hist["train_loss"], color=BLUE, lw=1.6, ls="-", label="Train")
ax[0].plot(e, hist["val_loss"], color=UNSEEN, lw=1.6, ls="--", label="Validation")
ax[0].fill_between(e, hist["train_loss"], hist["val_loss"], color=MUTED, alpha=0.12, lw=0,
                   label="Generalization gap")
# mark where val loss bottoms (early-stopping / best-checkpoint region)
bl = int(np.argmin(hist["val_loss"]))
ax[0].axvline(e[bl], color=INK2, lw=0.7, ls=":", zorder=1)
ax[0].text(e[bl] + 1.2, ax[0].get_ylim()[1] if False else 0.80, f"val-loss min\n(epoch {e[bl]})",
           fontsize=6.6, color=INK2, ha="left", va="top")
ax[0].set_title(f"(a) Training convergence — {TAG}")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
ax[0].legend(loc="upper right", fontsize=7.5); ax[0].set_xlim(1, len(e)); ax[0].set_ylim(0, None)

# (b) Dice / IoU
ax[1].plot(e, hist["val_dice"], color=BLUE, lw=1.6, ls="-", label="Dice")
ax[1].plot(e, hist["val_iou"], color=LBLUE, lw=1.6, ls="--", label="IoU")
bi = int(np.argmax(hist["val_dice"]))
ax[1].scatter([e[bi]], [hist["val_dice"][bi]], color=BLUE, s=22, zorder=6, edgecolor="white", lw=0.6)
# place label in the OPEN upper band, to the right of the point, clear of y-axis + legend
ax[1].annotate(f"best Dice = {hist['val_dice'][bi]:.3f} (epoch {e[bi]})",
               xy=(e[bi], hist["val_dice"][bi]), xytext=(e[bi] + 9, 0.975),
               ha="left", va="center", fontsize=7.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                               connectionstyle="arc3,rad=0.15"))
ax[1].set_title(f"(b) Validation Dice / IoU — {TAG}")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score")
ax[1].set_ylim(0.55, 1.0); ax[1].set_xlim(1, len(e)); ax[1].legend(loc="lower right", fontsize=7.5)

for a in ax:
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); full_box(a)
fig.tight_layout()
os.makedirs("outputs/figures_ssformer", exist_ok=True)
fig.savefig("outputs/figures_ssformer/Figure_training_SSFormerS.pdf", bbox_inches="tight")
fig.savefig("outputs/figures_ssformer/Figure_training_SSFormerS.png", dpi=320, bbox_inches="tight")
plt.close(fig); print("SAVED Figure_training_SSFormerS (annotation fixed)")
