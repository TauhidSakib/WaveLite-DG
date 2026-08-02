# Multi-sample visual interpretation of SSFormer-S (baseline), one image per test set.
# Columns: Original | Ground Truth | Prediction (Dice/IoU) | Grad-CAM overlay | Contour & edge analysis.
# Everything is REAL: TTA prediction, Seg-Grad-CAM from the decoder fusion layer, mask-derived contour metrics.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
from figlib import DEVICE, load_item, denorm, dice_of, SEEN
from ssformer_pld import SSFormerS
import matplotlib.pyplot as plt
from matplotlib import cm

sel = json.load(open("outputs/qual_selection.json"))          # same 5 images as the comparison figure

model = SSFormerS(1).to(DEVICE).eval()
model.load_state_dict(torch.load("outputs/baselines/ssformer_best.pt", map_location=DEVICE, weights_only=True))

# ---- hook the decoder fusion feature (256ch, 1/4 res) for Seg-Grad-CAM ----
_act, _grad = {}, {}
def _fhook(m, i, o):
    if not o.requires_grad:        # skip during no_grad TTA passes
        return
    _act["v"] = o
    o.register_hook(lambda g: _grad.__setitem__("v", g))
model.decode_head.linear_fuse1.register_forward_hook(_fhook)

@torch.no_grad()
def predict_tta(x):                                            # matches the reported SSFormer-S numbers
    outs = []
    for f in [lambda t: t, lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-2])]:
        outs.append(torch.sigmoid(f(model(f(x))[-1])))
    return torch.stack(outs).mean(0)[0, 0].cpu().numpy()

def gradcam(x):                                                # Seg-Grad-CAM: grad of foreground logits
    model.zero_grad(set_to_none=True)
    out = model(x)[-1]                                         # (1,1,H,W) logits
    score = out[out > 0].sum()                                 # sum of positive (foreground) logits
    score.backward()
    w = _grad["v"].mean(dim=(2, 3), keepdim=True)              # channel weights
    cam = F.relu((w * _act["v"]).sum(1, keepdim=True))         # (1,1,h,w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam -= cam.min(); cam /= (cam.max() + 1e-8)
    return cam

def iou_of(pr, gt, thr=0.5):
    p = pr > thr; t = gt > 0.5
    return float((p & t).sum() + 1) / float((p | t).sum() + 1)

def contour_stats(mask):
    """green boundary overlay + area(px) and circularity from the largest component."""
    m = mask > 0.5
    lbl, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(m), 0, 0.0
    big = (lbl == (1 + np.argmax(ndimage.sum(m, lbl, range(1, n + 1)))))
    area = int(big.sum())
    # digital perimeter = exposed 4-neighbour edges, scaled by pi/4 so a perfect disk -> circ 1.0
    hadj = int((big[:, :-1] & big[:, 1:]).sum()); vadj = int((big[:-1] & big[1:]).sum())
    perim = (4 * area - 2 * hadj - 2 * vadj) * (np.pi / 4)
    circ = float(4 * np.pi * area / (perim ** 2 + 1e-8)) if perim else 0.0
    bound = big & ~ndimage.binary_erosion(big, iterations=1)   # 1-px boundary
    ring = ndimage.binary_dilation(bound, iterations=2)        # thicken for display
    return ring, area, min(circ, 1.0)

RED = "#C0392B"; GREEN = np.array([0.16, 0.85, 0.30]); JET = plt.get_cmap("jet")
nrow, ncol = len(sel), 5
fig, ax = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.28 * nrow))
heads = ["Original\nEndoscopy Image", "Ground Truth\nAnnotation Mask", "Predicted\nSegmentation Mask",
         "Grad-CAM\nActivation Overlay", "Contour Analysis\n& Edge Detection"]
HEADC = "#1a7a3c"

log = []
for r, s in enumerate(sel):
    it, mt = load_item(s["img"], s["mask"])
    x = it.unsqueeze(0).to(DEVICE)
    img = denorm(it); gt = mt[0].numpy()
    pr = predict_tta(x); pb = (pr > 0.5).astype(float)
    d = dice_of(pr, gt); iou = iou_of(pr, gt)
    cam = gradcam(x.clone().requires_grad_(True))
    ring, area, circ = contour_stats(pb)

    # 0 original / 1 GT / 2 prediction
    ax[r, 0].imshow(img)
    ax[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax[r, 2].imshow(pb, cmap="gray", vmin=0, vmax=1)
    ax[r, 2].text(0.03, 0.97, f"Dice={d:.3f}\nIoU={iou:.3f}", transform=ax[r, 2].transAxes,
                  ha="left", va="top", fontsize=7.5, color="white", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec="none"))
    # 3 Grad-CAM overlay
    heat = JET(cam)[..., :3]
    ax[r, 3].imshow(0.45 * img + 0.55 * heat)
    # 4 contour + edge
    edges = ndimage.sobel(img.mean(2))**2
    edges = ndimage.sobel(img.mean(2), axis=0)**2 + ndimage.sobel(img.mean(2), axis=1)**2
    edges = np.clip(edges / (np.percentile(edges, 99) + 1e-8), 0, 1)
    base = np.clip(img + 0.20 * edges[..., None], 0, 1)         # subtle edge emphasis
    over = base.copy(); over[ring > 0] = GREEN
    ax[r, 4].imshow(over)
    ax[r, 4].text(0.03, 0.97, f"Area={area}px$^2$\nCirc={circ:.2f}", transform=ax[r, 4].transAxes,
                  ha="left", va="top", fontsize=7.5, color="white", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc=(0.10, 0.55, 0.20, 0.90), ec="none"))

    ax[r, 0].set_ylabel(f"{s['dataset']}\n({s['kind']})", fontsize=8.5,
                        color=(SEEN if s['kind'] == 'seen' else '#B4611A'), fontweight="bold", labelpad=5)
    for c in range(ncol):
        a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color("#cccccc"); sp.set_linewidth(0.6)
        if r == 0:
            a.set_title(heads[c], fontsize=9, fontweight="bold", color=HEADC, pad=6)
    log.append((s["dataset"], d, iou, area, circ))

fig.suptitle("Multi-Sample Visual Interpretation of Polyp Segmentation — SSFormer-S (baseline)",
             fontsize=11.5, fontweight="bold", y=0.998)
fig.tight_layout(rect=[0, 0.005, 1, 0.975]); fig.subplots_adjust(hspace=0.16, wspace=0.05)
os.makedirs("outputs/figures_ssformer", exist_ok=True)
fig.savefig("outputs/figures_ssformer/Figure_SSFormerS_interpretation.pdf", bbox_inches="tight")
fig.savefig("outputs/figures_ssformer/Figure_SSFormerS_interpretation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("SAVED Figure_SSFormerS_interpretation")
for name, d, iou, area, circ in log:
    print(f"  {name:20s} Dice={d:.3f} IoU={iou:.3f} area={area}px circ={circ:.2f}")
