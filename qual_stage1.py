# Stage 1: pick 1 representative (median polyp-area) image per dataset [MODEL-AGNOSTIC selection],
# run the 4 native models (ours/pranet/polyppvt/ssformer) with TTA, cache inputs/GT/preds/Dice.
# VM-UNet is done separately in WSL (qual_stage2_wsl.py) on the SAME selected images.
import os, json, sys
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch
from figlib import *
from train_baseline import build, tta_prob   # reuse exact inference path
sys.path.insert(0, "_baselines/PraNet"); sys.path.insert(0, "_baselines/Polyp-PVT")

DSETS = ["Kvasir-SEG","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]
KIND  = {"Kvasir-SEG":"seen","CVC-ClinicDB":"seen","CVC-ColonDB":"unseen","ETIS-LaribPolypDB":"unseen","CVC-300":"unseen"}

# ---- model-agnostic selection: per dataset, the image whose GT area == dataset median area ----
sel = []
for d in DSETS:
    pairs = list_pairs(*TEST_SETS[d][1:])
    areas = []
    for ip, mp in pairs:
        it, mt = load_item(ip, mp); g = mt[0].numpy() > 0.5
        areas.append(g.mean())
    areas = np.array(areas)
    valid = areas > 0.002
    idx = np.where(valid)[0]
    med = np.median(areas[valid])
    j = idx[np.argmin(np.abs(areas[idx] - med))]   # closest-to-median, deterministic
    sel.append({"dataset": d, "kind": KIND[d], "img": str(pairs[j][0]), "mask": str(pairs[j][1]),
                "area_pct": round(float(areas[j]*100), 2)})
    print(f"{d:20s} n={len(pairs)} median-area pick area={areas[j]*100:.2f}%  {os.path.basename(pairs[j][0])}")
json.dump(sel, open("outputs/qual_selection.json", "w"), indent=1)

# ---- cache denormalized input + GT once ----
inputs, gts = [], []
for s in sel:
    it, mt = load_item(s["img"], s["mask"])
    inputs.append(denorm(it)); gts.append((mt[0].numpy() > 0.5).astype(np.uint8))
inputs = np.stack(inputs); gts = np.stack(gts)

# NOTE: build ssformer BEFORE polyppvt — importing Polyp-PVT registers a clashing
# 'pvt_v2_b2' into timm's registry that otherwise shadows SSFormer's real backbone.
MODELS = [("ours","outputs/wavemamba_dg_best.pt"),
          ("pranet","outputs/baselines/pranet_best.pt"),
          ("ssformer","outputs/baselines/ssformer_best.pt"),
          ("polyppvt","outputs/baselines/polyppvt_best.pt")]
preds = {}; dice = {}
for key, ckpt in MODELS:
    if key == "ours":
        m = WaveMambaDG(pretrained=False).to(DEVICE)
    else:
        m = build(key).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True)); m.eval()
    pk, dk = [], []
    for s, gt in zip(sel, gts):
        it, mt = load_item(s["img"], s["mask"]); pr = tta_prob(m, it)
        pb = (pr > 0.5).astype(np.uint8); g = gt.astype(np.float64)
        inter = (pb*g).sum(); dc = (2*inter + 1) / (pb.sum() + g.sum() + 1)
        pk.append(pb); dk.append(float(dc))
    preds[key] = np.stack(pk); dice[key] = dk
    print(f"{key:10s} Dice/row = " + ", ".join(f"{v:.3f}" for v in dk))
    del m; torch.cuda.empty_cache()

np.savez_compressed("outputs/qual_cache.npz", inputs=inputs, gts=gts,
                    **{f"pred_{k}": v for k, v in preds.items()})
json.dump(dice, open("outputs/qual_dice_native.json", "w"), indent=1)
print("STAGE1_DONE")
