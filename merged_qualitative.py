# Merged qualitative figure for SSFormer-S (baseline): input | GT | prediction | error map | Grad-CAM.
# Reuses the cached per-image Dice selection (no re-scoring). Real forward passes only.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch
from figlib import DEVICE, load_item, denorm
from ssformer_pld import SSFormerS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "serif", "font.size": 8})
EPS = 1e-6

model = SSFormerS(1).to(DEVICE).eval()
model.load_state_dict(torch.load("outputs/baselines/ssformer_best.pt", map_location=DEVICE, weights_only=True))

# --- Grad-CAM hook: PVTv2 backbone stride-32 stage (deepest, 512ch, 11x11) ---
STAGE = 3
_acts, _grads = {}, {}
def _bhook(module, inp, out):
    if not out[STAGE].requires_grad:                 # skip no_grad TTA
        return
    _acts["A"] = out[STAGE]
    out[STAGE].register_hook(lambda g: _grads.__setitem__("g", g))
model.backbone.register_forward_hook(_bhook)

@torch.no_grad()
def predict_tta(x):
    outs = []
    for f in [lambda t: t, lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-2])]:
        outs.append(torch.sigmoid(f(model(f(x))[-1])))
    return torch.stack(outs).mean(0)[0, 0].cpu().numpy()

def dice_iou(pr, gt, thr=0.5, eps=EPS):
    p = (pr > thr).astype(np.float64); t = (gt > 0.5).astype(np.float64)
    I = (p * t).sum(); Pp = p.sum(); T = t.sum(); U = Pp + T - I
    return float((2 * I + eps) / (Pp + T + eps)), float((I + eps) / (U + eps))

def gradcam32(x):
    _acts.clear(); _grads.clear(); model.zero_grad(set_to_none=True)
    out = model(x)[-1]
    fg = (torch.sigmoid(out) > 0.5).float()
    ((out * fg).sum() / fg.sum().clamp(min=1.0)).backward()
    A, g = _acts["A"], _grads["g"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(1, keepdim=True))[0, 0].detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)                  # per-panel min-max, native 11x11

# --- selection from cache (same 5 median + 2 failure) ---
per = json.load(open("outputs/ssformer_gradcam_perimage.json"))
KIND = {"Kvasir-SEG": "seen", "CVC-ClinicDB": "seen", "CVC-ColonDB": "unseen",
        "ETIS-LaribPolypDB": "unseen", "CVC-300": "unseen"}
def pick(name, q):
    rows = per[name]; ds = np.array([r[2] for r in rows])
    tgt = np.median(ds) if q == "median" else np.percentile(ds, 10)
    return rows[int(np.argmin(np.abs(ds - tgt)))]
SEL = [(n, KIND[n], "median", *pick(n, "median")[:2]) for n in per]
SEL += [(n, "unseen", "failure", *pick(n, "p10")[:2]) for n in ("CVC-ColonDB", "ETIS-LaribPolypDB")]

# --- colours (error map): TP green, FP red, FN blue -- only these three ---
TP = np.array([0.16, 0.70, 0.26]); FP = np.array([0.86, 0.16, 0.16]); FN = np.array([0.16, 0.36, 0.92])
RED = "#C0392B"; JET = plt.get_cmap("jet"); GREEN = "#1a7a3c"

# LANDSCAPE: 5 panel-type rows (a-e) x 7 case columns
nrow, ncol = 5, len(SEL)
fig, ax = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.18 * nrow))
rlabels = ["(a) Input\nEndoscopy Image", "(b) Ground Truth\nAnnotation Mask", "(c) Predicted\nSegmentation Mask",
           "(d) Error Map\n(TP / FP / FN)", "(e) Grad-CAM\n(stride-32 bottleneck)"]

for c, (name, kind, tag, ip, mp) in enumerate(SEL):
    it, mt = load_item(ip, mp)
    x = it.unsqueeze(0).to(DEVICE)
    img = denorm(it); gt = mt[0].numpy()
    pr = predict_tta(x); pb = pr > 0.5; gb = gt > 0.5
    d, iou = dice_iou(pr, gt)
    cam = gradcam32(x); H, W = img.shape[:2]

    ax[0, c].imshow(img)
    ax[1, c].imshow(gb.astype(float), cmap="gray", vmin=0, vmax=1)
    ax[2, c].imshow(pb.astype(float), cmap="gray", vmin=0, vmax=1)
    badge = RED if tag == "median" else "#7a1f1f"
    ax[2, c].text(0.03, 0.97, f"Dice={d:.3f}\nIoU={iou:.3f}", transform=ax[2, c].transAxes,
                  ha="left", va="top", fontsize=7.0, color="white", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.22", fc=badge, ec="none"))
    # (d) error map: blend the three region colours onto the input; no other overlay
    err = img.copy()
    for mask, col in [(gb & pb, TP), ((~gb) & pb, FP), (gb & (~pb), FN)]:
        err[mask] = 0.45 * err[mask] + 0.55 * col
    ax[3, c].imshow(np.clip(err, 0, 1))
    # (e) Grad-CAM: native 11x11, nearest, per-panel normalized
    a = ax[4, c]; a.imshow(img, extent=[0, W, H, 0])
    a.imshow(cam, cmap=JET, alpha=0.5, extent=[0, W, H, 0], interpolation="nearest", vmin=0, vmax=1)
    a.set_xlim(0, W); a.set_ylim(H, 0)

    ctitle = f"{name}\n({kind})" if tag == "median" else f"{name}\n(unseen, failure)"
    ccol = ("#2b6cb0" if kind == "seen" else "#B4611A") if tag == "median" else RED
    ax[0, c].set_title(ctitle, fontsize=8.2, color=ccol, fontweight="bold", pad=6)
    for r in range(nrow):
        a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#bbb"); s.set_linewidth(0.6)
        if c == 0:
            a.set_ylabel(rlabels[r], fontsize=8.2, fontweight="bold", color=GREEN, labelpad=5)

fig.suptitle("Qualitative Interpretation of Polyp Segmentation — SSFormer-S (baseline)",
             fontsize=12, fontweight="bold", y=0.995)
# legend for the error map (only the three colours)
fig.legend(handles=[Patch(fc=TP, label="True positive"), Patch(fc=FP, label="False positive"),
                    Patch(fc=FN, label="False negative")],
           loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.033))
# kept Grad-CAM caption note (normalization + resolution + finding)
fig.text(0.5, 0.012,
         "Grad-CAM: each map independently min–max normalized to [0,1] (jet, α=0.5), not comparable across rows; "
         "native 11×11 resolution (nearest-neighbour, no upsampling/smoothing). "
         "On the two most domain-shifted sets (CVC-ColonDB, ETIS) the bottleneck evidence sits off the polyp despite accurate masks.",
         ha="center", fontsize=6.5, style="italic", color="#333")
fig.tight_layout(rect=[0, 0.052, 1, 0.984]); fig.subplots_adjust(hspace=0.13, wspace=0.05)
for ext in ("pdf", "png"):
    fig.savefig(fr"D:\Downloads\merged_qualitative_figure.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("SAVED D:\\Downloads\\merged_qualitative_figure.pdf / .png")
for name, kind, tag, ip, mp in SEL:
    print(f"  {tag:7s} {name:20s} {kind:6s} {os.path.basename(ip)}")
