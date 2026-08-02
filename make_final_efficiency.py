# Honest 2-panel efficiency trade-off: mean-unseen Dice vs params, and vs single-pass FPS.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np
from figlib import *
import matplotlib.pyplot as plt

eff = json.load(open("outputs/efficiency.json"))
OUTB="outputs/baselines"
MODELS=[("WaveLite-DG","ours"),("U-Net","unet"),("PraNet","pranet"),
        ("Polyp-PVT","polyppvt"),("SSFormer-S","ssformer"),("VM-UNet","vmunet")]
UNSEEN=["CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]
def macro(key):
    r=json.load(open(f"{OUTB}/{key}_report.json"))["report"]
    return float(np.mean([r[d][1] for d in UNSEEN]))
MCOL={"WaveLite-DG":BLUE,"U-Net":"#8a8a8a","PraNet":LGREEN,"Polyp-PVT":ORNG,"SSFormer-S":LBLUE,"VM-UNet":MAG}
PFALL={"vmunet":44.27}  # params for models whose FPS is not yet measured
def pm(key): return eff[key]["params_M"] if key in eff and "params_M" in eff[key] else PFALL.get(key)

os.makedirs("outputs/figures_final", exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(8.0,3.5))
# (a) Dice vs params
for disp,key in MODELS:
    y=macro(key); x=pm(key); hl=disp=="WaveLite-DG"; right=x>27
    if x is None: continue
    ax[0].scatter(x,y,s=180 if hl else 85,color=MCOL[disp],edgecolor="black",lw=1.2 if hl else 0.6,
                  marker="*" if hl else "o",zorder=5)
    if y<0.75:   # bottom cluster (U-Net, PraNet) — label below to avoid horizontal collision
        ax[0].annotate(disp,(x,y),textcoords="offset points",xytext=(0,-13),ha="center",va="top",
                       fontsize=7.2,color=INK2)
    else:
        ax[0].annotate(disp+(" (ours)" if hl else ""),(x,y),textcoords="offset points",
                       xytext=(-9 if right else 9,5),ha="right" if right else "left",
                       fontsize=7.2,fontweight="bold" if hl else "normal",color=INK if hl else INK2)
ax[0].set_xlabel("Parameters (M)  ← smaller is better"); ax[0].set_ylabel("Mean unseen Dice")
ax[0].set_xlim(3,48); ax[0].set_ylim(0.725,0.845); ax[0].set_title("(a) Accuracy vs. model size")
ax[0].grid(color=GRID,lw=0.6); ax[0].set_axisbelow(True); full_box(ax[0])
# (b) Dice vs FPS (single-pass); vmunet FPS pending
for disp,key in MODELS:
    if "fps_single" not in eff.get(key,{}) or not isinstance(eff[key].get("fps_single"),(int,float)): continue
    y=macro(key); x=eff[key]["fps_single"]; hl=disp=="WaveLite-DG"; right=x>60
    ax[1].scatter(x,y,s=180 if hl else 85,color=MCOL[disp],edgecolor="black",lw=1.2 if hl else 0.6,
                  marker="*" if hl else "o",zorder=5)
    ax[1].annotate(disp+(" (ours)" if hl else ""),(x,y),textcoords="offset points",
                   xytext=(-9 if right else 9,5),ha="right" if right else "left",
                   fontsize=7.2,fontweight="bold" if hl else "normal",color=INK if hl else INK2)
ax[1].set_xlabel("Inference speed (FPS, single-pass)  → faster is better"); ax[1].set_ylabel("Mean unseen Dice")
ax[1].set_ylim(0.725,0.845); ax[1].set_xlim(0,100)
ax[1].set_title("(b) Accuracy vs. speed"); ax[1].grid(color=GRID,lw=0.6); ax[1].set_axisbelow(True); full_box(ax[1])
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(f"outputs/figures_final/Figure_efficiency.{ext}", dpi=320, bbox_inches="tight")
plt.close(fig)
print("SAVED outputs/figures_final/Figure_efficiency.{pdf,png}")
print("params/FPS/Dice:")
for disp,key in MODELS:
    e=eff.get(key,{})
    print(f"  {disp:14s} params {pm(key)}  fps {e.get('fps_single','pending')}  meanDice {macro(key):.4f}")
