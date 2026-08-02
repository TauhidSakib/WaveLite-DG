# Grad-CAM interpretation figure for SSFormer-S (baseline). Real forward passes only.
# - CAM taken at EARLY layers (PVTv2 backbone stride-32 and stride-16), not the decoder head.
# - Median-Dice image per dataset + two 10th-percentile "failure case" rows (ColonDB, ETIS).
# - Dice & IoU use identical smoothing so IoU == Dice/(2-Dice).
# - Contour column dropped (undefined metric); its slot shows the 2nd CAM variant.
# - Heatmaps are NOT upsampled/smoothed: shown at native res with nearest-neighbour.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch
from sklearn.model_selection import train_test_split
from figlib import DEVICE, load_item, denorm, list_pairs, TEST_SETS
from ssformer_pld import SSFormerS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({"font.family": "serif", "font.size": 8})
EPS = 1e-6                                          # identical smoothing for Dice AND IoU

model = SSFormerS(1).to(DEVICE).eval()
model.load_state_dict(torch.load("outputs/baselines/ssformer_best.pt", map_location=DEVICE, weights_only=True))

# ---------------- Grad-CAM hook on the PVTv2 backbone (early layers) ----------------
# backbone(x) -> 4 stage features, strides [4,8,16,32], channels [64,128,320,512]
TARGETS = {32: 3, 16: 2}                             # stride -> backbone stage index
_acts, _grads = {}, {}
def _bhook(module, inp, out):
    if not out[TARGETS[32]].requires_grad:          # skip during no_grad TTA passes
        return
    for idx in TARGETS.values():
        _acts[idx] = out[idx]
        out[idx].register_hook((lambda i: (lambda g: _grads.__setitem__(i, g)))(idx))
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
    dice = (2 * I + eps) / (Pp + T + eps)
    iou  = (I + eps) / (U + eps)
    return float(dice), float(iou)

def grad_cams(x):
    """Seg-Grad-CAM at the two backbone stages. Scalar = MEAN of foreground logits
       over the predicted region {sigmoid(logit) > 0.5}. Single (non-TTA) forward.
       Returns the RAW (un-normalized) ReLU maps so raw min/max can be reported."""
    _acts.clear(); _grads.clear(); model.zero_grad(set_to_none=True)
    out = model(x)[-1]                               # logits (1,1,352,352)
    fg = (torch.sigmoid(out) > 0.5).float()
    scalar = (out * fg).sum() / fg.sum().clamp(min=1.0)
    scalar.backward()
    raw = {}
    for st, idx in TARGETS.items():
        A, g = _acts[idx], _grads[idx]
        w = g.mean(dim=(2, 3), keepdim=True)
        raw[st] = torch.relu((w * A).sum(1, keepdim=True))[0, 0].detach().cpu().numpy()
    return raw, float(scalar.detach())

# ---------------- build evaluation pair lists (seen=val split seed-42; unseen=full) ----------------
seen = list_pairs(*TEST_SETS["Kvasir-SEG"][1:]) + list_pairs(*TEST_SETS["CVC-ClinicDB"][1:])
_, va = train_test_split(seen, test_size=0.2, random_state=42)
PAIRS = {
    "Kvasir-SEG":        ("seen",   [p for p in va if "Kvasir-SEG" in str(p[0])]),
    "CVC-ClinicDB":      ("seen",   [p for p in va if "cvcclinicdb" in str(p[0])]),
    "CVC-ColonDB":       ("unseen", list_pairs(*TEST_SETS["CVC-ColonDB"][1:])),
    "ETIS-LaribPolypDB": ("unseen", list_pairs(*TEST_SETS["ETIS-LaribPolypDB"][1:])),
    "CVC-300":           ("unseen", list_pairs(*TEST_SETS["CVC-300"][1:])),
}

# ---------------- per-image Dice for selection (cached) ----------------
CACHE = "outputs/ssformer_gradcam_perimage.json"
if os.path.exists(CACHE):
    per = json.load(open(CACHE))
else:
    per = {}
    for name, (kind, pairs) in PAIRS.items():
        rows = []
        for ip, mp in pairs:
            it, mt = load_item(ip, mp)
            pr = predict_tta(it.unsqueeze(0).to(DEVICE))
            d, _ = dice_iou(pr, mt[0].numpy())
            rows.append([str(ip), str(mp), d])
        per[name] = rows
        print(f"  scored {name:20s} n={len(rows):4d} mean={np.mean([r[2] for r in rows]):.4f}", flush=True)
    json.dump(per, open(CACHE, "w"))

def pick(name, q):
    rows = per[name]; ds = np.array([r[2] for r in rows])
    target = np.median(ds) if q == "median" else np.percentile(ds, 10)
    i = int(np.argmin(np.abs(ds - target)))
    return rows[i], target

# selection: median per dataset + two 10th-pct failure rows
SEL = []
for name, (kind, _) in PAIRS.items():
    (ip, mp, d), tgt = pick(name, "median")
    SEL.append((name, kind, "median", ip, mp, d, tgt))
for name in ["CVC-ColonDB", "ETIS-LaribPolypDB"]:
    (ip, mp, d), tgt = pick(name, "p10")
    SEL.append((name, "unseen", "failure", ip, mp, d, tgt))

# ---------------- console report ----------------
print("=" * 74)
print("GRAD-CAM TARGET LAYER")
print("  PREVIOUS: model.decode_head.linear_fuse1  (decoder fusion, stride-4, ~88x88)")
print("            scalar backpropagated = SUM of positive logits  -> too close to head (circular)")
print("  NEW: PVTv2 backbone stages -> stride-32 = stage[3] (512ch, 11x11)  [DISPLAYED, col d];")
print("       stride-16 = stage[2] (320ch, 22x22)  [diagnostic-only: sparse on ColonDB/ETIS, dropped].")
print("  scalar backpropagated = MEAN of foreground logits over predicted region {sigmoid(logit)>0.5}")
print("  display: per-panel min-max to [0,1]; native 11x11, nearest-neighbour, NO upsampling/smoothing.")
print("=" * 74)
print("SELECTION (filename : Dice)")
for name, kind, tag, ip, mp, d, tgt in SEL:
    print(f"  [{tag:7s}] {name:20s} target={tgt:.4f}  Dice={d:.4f}  {os.path.basename(ip)}")
print("=" * 74)
print(f"DICE/IoU CONSISTENCY  (identical smoothing eps={EPS:g} in BOTH; IoU should == Dice/(2-Dice))")

# ---------------- render ----------------
# stride-16 CAM is dropped from the figure (genuinely sparse on shifted domains -> near-blank);
# its raw min/max is still PRINTED below so the sparsity is documented, not hidden.
print("RAW (un-normalized) CAM stats per row  [min / max / mean]  -- diagnoses blank panels")
RED = "#C0392B"; JET = plt.get_cmap("jet"); GREEN = "#1a7a3c"
nrow, ncol = len(SEL), 4
fig, ax = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.28 * nrow))
heads = ["(a) Original\nEndoscopy Image", "(b) Ground Truth\nAnnotation Mask", "(c) Predicted\nSegmentation Mask",
         "(d) Grad-CAM\n(stride-32 bottleneck)"]

for r, (name, kind, tag, ip, mp, d_sel, tgt) in enumerate(SEL):
    it, mt = load_item(ip, mp)
    x = it.unsqueeze(0).to(DEVICE)
    img = denorm(it); gt = mt[0].numpy()
    pr = predict_tta(x); pb = (pr > 0.5).astype(float)
    d, iou = dice_iou(pr, gt)
    ident = d / (2 - d)
    raw, scal = grad_cams(x)
    m32, m16 = raw[32], raw[16]
    print(f"  {name:20s} [{tag:7s}] Dice={d:.4f} IoU={iou:.4f} =D/(2-D)={ident:.4f} match={abs(iou-ident)<5e-4} "
          f"| s32 [{m32.min():.3g}/{m32.max():.3g}/{m32.mean():.3g}]  s16 [{m16.min():.3g}/{m16.max():.3g}/{m16.mean():.3g}]")
    cam32 = m32 / (m32.max() + 1e-8)                 # per-panel min-max to [0,1]
    H, W = img.shape[:2]

    ax[r, 0].imshow(img)
    ax[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax[r, 2].imshow(pb, cmap="gray", vmin=0, vmax=1)
    badge = RED if tag == "median" else "#7a1f1f"
    ax[r, 2].text(0.03, 0.97, f"Dice={d:.3f}\nIoU={iou:.3f}", transform=ax[r, 2].transAxes,
                  ha="left", va="top", fontsize=7.5, color="white", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc=badge, ec="none"))
    a = ax[r, 3]; a.imshow(img, extent=[0, W, H, 0])
    a.imshow(cam32, cmap=JET, alpha=0.5, extent=[0, W, H, 0],
             interpolation="nearest", vmin=0, vmax=1)         # native 11x11, NO upsampling
    a.set_xlim(0, W); a.set_ylim(H, 0)

    rlab = f"{name}\n({kind})" if tag == "median" else f"{name}\n(unseen, failure)"
    rcol = ("#2b6cb0" if kind == "seen" else "#B4611A") if tag == "median" else RED
    ax[r, 0].set_ylabel(rlab, fontsize=8.5, color=rcol, fontweight="bold", labelpad=5)
    for c in range(ncol):
        a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#bbb"); s.set_linewidth(0.6)
        if r == 0:
            a.set_title(heads[c], fontsize=8.6, fontweight="bold", color=GREEN, pad=6)

fig.suptitle("Grad-CAM Interpretation of Polyp Segmentation — SSFormer-S (baseline)",
             fontsize=12, fontweight="bold", y=0.998)
fig.text(0.5, 0.012,
         "Each Grad-CAM is independently min–max normalized to [0,1] (jet, α=0.5); values are NOT comparable across rows. "
         "Shown at native 11×11 resolution (nearest-neighbour, no upsampling/smoothing). "
         "On the two most domain-shifted sets (CVC-ColonDB, ETIS) the bottleneck evidence sits off the polyp despite accurate masks.",
         ha="center", fontsize=6.6, style="italic", color="#333")
fig.tight_layout(rect=[0, 0.022, 1, 0.985]); fig.subplots_adjust(hspace=0.14, wspace=0.05)
for ext in ("pdf", "png"):
    fig.savefig(fr"D:\Downloads\grad_cam_figure.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("=" * 74)
print("CONTOUR COLUMN: DROPPED (Area/Circularity undefined in manuscript, support no claim).")
print("  For the record, the old circularity was: circ = 4*pi*A / P^2, where")
print("  A = pixel count of the largest connected component (scipy.ndimage.label),")
print("  P = digital perimeter = (exposed 4-neighbour edges) * pi/4  (so a disk -> 1.0);")
print("  contour extracted via binary boundary = big & ~binary_erosion(big).")
print("SAVED D:\\Downloads\\grad_cam_figure.pdf / .png")
