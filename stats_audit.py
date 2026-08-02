# Rigorous statistical re-analysis from the real per-image Dice (no retraining).
# Per-dataset + pooled Wilcoxon (two-sided), bootstrap 95% CI of mean paired diff,
# effect sizes (Cohen's d_z, rank-biserial), Holm correction, macro vs pooled means.
import os, json, numpy as np
os.chdir(r"D:\AI-Projects\WaveMamba")
from scipy.stats import wilcoxon
rng = np.random.default_rng(0)

OUTB = "outputs/baselines"
MODELS = ["ours","unet","pranet","polyppvt","ssformer","vmunet"]
NAME = {"ours":"WaveLite-DG","unet":"U-Net","pranet":"PraNet","polyppvt":"Polyp-PVT","ssformer":"SSFormer-S","vmunet":"VM-UNet"}
UNSEEN = ["CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]

PI = {m: json.load(open(f"{OUTB}/{m}_perimage.json")) for m in MODELS}

# --- alignment / integrity checks ---
print("=== ALIGNMENT CHECK (per-image counts per unseen dataset) ===")
for m in MODELS:
    lens = {d: len(PI[m][d]) for d in UNSEEN}
    print(f"  {NAME[m]:14s} {lens}  pooled={sum(lens.values())}")
# all models must have identical lengths per dataset (=> position i is the same sorted image for all)
ref = {d: len(PI['ours'][d]) for d in UNSEEN}
ok = all(all(len(PI[m][d])==ref[d] for d in UNSEEN) for m in MODELS)
print("  all lengths match across models:", ok, "(=> per-position pairing == per-filename pairing, since list_pairs sorts by name)")

def pool(m): return np.concatenate([np.array(PI[m][d]) for d in UNSEEN])

print("\n=== MEAN DICE: dataset-macro vs pooled-per-image ===")
for m in MODELS:
    macro = np.mean([np.mean(PI[m][d]) for d in UNSEEN])
    pooled = np.mean(pool(m))
    print(f"  {NAME[m]:14s} macro={macro:.4f}  pooled(image-weighted)={pooled:.4f}")

def boot_ci(diff, n=10000):
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def analyze(a, b):
    d = a - b
    mean_d = float(d.mean()); med_d = float(np.median(d))
    dz = mean_d/(d.std(ddof=1)+1e-12)
    try:
        W, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except Exception:
        W, p = np.nan, np.nan
    # rank-biserial for paired wilcoxon
    nz = d[d!=0];
    if len(nz):
        r = np.argsort(np.argsort(np.abs(nz)))+1
        Rpos = r[nz>0].sum(); Rneg = r[nz<0].sum(); rb = (Rpos-Rneg)/(Rpos+Rneg)
    else: rb = 0.0
    lo, hi = boot_ci(d)
    return mean_d, med_d, dz, rb, p, lo, hi

print("\n=== PAIRED TESTS: WaveLite-DG (ours) vs each baseline ===")
print("  scope         baseline      meanΔ    medΔ     d_z    rank-bis   p(2-sided)   95%CI(meanΔ)")
ours_pool = pool("ours")
pooled_p = {}
for m in MODELS:
    if m=="ours": continue
    for scope in UNSEEN+["POOLED"]:
        if scope=="POOLED":
            a, b = ours_pool, pool(m)
        else:
            a, b = np.array(PI["ours"][scope]), np.array(PI[m][scope])
        md, medd, dz, rb, p, lo, hi = analyze(a, b)
        if scope=="POOLED": pooled_p[m]=p
        sc = scope if scope!="POOLED" else "POOLED"
        print(f"  {sc:13s} {NAME[m]:12s} {md:+.4f}  {medd:+.4f}  {dz:+.3f}   {rb:+.3f}   {p:.2e}   [{lo:+.4f},{hi:+.4f}]")
    print()

# Holm correction across the 4 pooled comparisons
print("=== HOLM correction (pooled two-sided p, 4 comparisons) ===")
items = sorted(pooled_p.items(), key=lambda kv: kv[1]); k=len(items)
for i,(m,p) in enumerate(items):
    padj = min(1.0, p*(k-i))
    sig = "sig" if padj<0.05 else "n.s."
    print(f"  {NAME[m]:14s} raw p={p:.2e}  Holm-adj={padj:.2e}  {sig}")
print("\nDONE_STATS")
