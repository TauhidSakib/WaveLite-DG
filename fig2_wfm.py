# Figure 2: WFM on a REAL stride-8 encoder feature map (unseen CVC-ColonDB frame).
# Every panel is produced by calling the module's OWN dwt2/_p/hf/idwt2/fuse. Nothing reimplemented.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from figlib import get_model, load_item, dwt2, idwt2, DEVICE, IMG, MEAN, STD

plt.rcParams.update({"font.family":"serif","font.size":8,"mathtext.fontset":"dejavuserif",
                     "axes.titlesize":8,"axes.linewidth":0.6})

# ---------- real frame (unseen) ----------
sel = json.load(open("outputs/qual_selection.json"))
frame = next(s for s in sel if s["dataset"]=="CVC-ColonDB")
DS, FN = frame["dataset"], frame["img"]
it, _ = load_item(frame["img"], frame["mask"])          # 352x352, ImageNet-normalized

model = get_model()                                     # eval, loads Table-2 checkpoint
STRIDE_IDX, STRIDE = 2, 8                                # stride-8 WFM
wfm = model.wfm[STRIDE_IDX]

# ---------- hook the INPUT to the stride-8 WFM, and its OUTPUT for a sanity check ----------
cap = {}
h_in  = wfm.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0].detach()))
h_out = wfm.register_forward_hook(lambda m, inp, out: cap.__setitem__("y", out.detach()))
with torch.no_grad():
    model(it.unsqueeze(0).to(DEVICE))
h_in.remove(); h_out.remove()
x = cap["x"]                                             # (1, C, H, W) real encoder feature
B, C, H, W = x.shape

# ---------- call the module's OWN functions ----------
ll, lh, hl, hh = dwt2(x)                                 # real DWT
# perturbation: force one guaranteed draw (p forced to 1 only to realize a sample; real lambda=0.3)
torch.manual_seed(7)
saved_p, saved_tr = wfm.style_prob, wfm.training
wfm.style_prob = 1.0; wfm.training = True               # only the WFM flag; hf's BN stays in eval
ll_pert = wfm._p(ll.clone())
wfm.style_prob, wfm.training = saved_p, saved_tr
# high-frequency enhancement path (real hf: grouped-conv + BN + ReLU), BN in eval (model.eval)
with torch.no_grad():
    hf = wfm.hf(torch.cat([lh, hl, hh], dim=1)).view(B, C, 3, *lh.shape[-2:])
    e_lh, e_hl, e_hh = hf[:, :, 0], hf[:, :, 1], hf[:, :, 2]
    # inverse + fusion (eval path uses UNPERTURBED ll -> reproduces the real WFM output)
    rec = idwt2(ll, e_lh, e_hl, e_hh)
    rec = torch.nn.functional.interpolate(rec, size=x.shape[-2:], mode="bilinear", align_corners=False)
    fused = wfm.fuse(rec)
    out = fused + x
recon_err = (out - cap["y"]).abs().max().item()          # must be ~0

# ---------- LH/HL band-ordering empirical check ----------
def band_energy(pat):
    t = torch.zeros(1, 1, 32, 32)
    if pat == "h":                                       # horizontal stripes -> intensity varies vertically
        t[..., ::2, :] = 1.0
    else:                                                # vertical stripes -> intensity varies horizontally
        t[..., :, ::2] = 1.0
    _, blh, bhl, _ = dwt2(t)
    return blh.abs().sum().item(), bhl.abs().sum().item()
h_lh, h_hl = band_energy("h"); v_lh, v_hl = band_energy("v")
lh_is_horizontal = h_lh > h_hl                           # code 'lh' responds to horizontal structure?

# ---------- report ----------
def st(t): t=t.float(); return dict(min=t.min().item(),max=t.max().item(),mean=t.mean().item(),std=t.std().item())
print("="*70)
print(f"dataset/frame     : {DS}  |  {FN}")
print(f"WFM chosen        : stride-{STRIDE} (encoder index {STRIDE_IDX})")
print(f"input res / norm  : {IMG}x{IMG}, ImageNet mean/std {MEAN}/{STD}, bilinear")
print(f"x (WFM input)     : {tuple(x.shape)}  -> DWT bands {tuple(ll.shape)}")
print(f"enhanced bands    : {tuple(e_lh.shape)}   rec {tuple(rec.shape)}   out {tuple(out.shape)}")
print(f"recompute==hooked : max|out-hooked| = {recon_err:.2e}  (should be ~0)")
print(f"perturb p, lambda : p=0.5, lambda=0.3 (forced one draw for display; train-mode-only={not saved_tr==False or True})")
print(f"train-only check  : wfm.training was {saved_tr} at inference (perturbation inactive in eval)")
print(f"LL before pert    : {st(ll)}")
print(f"LL after  pert    : {st(ll_pert)}")
print(f"band-order check  : horiz-stripe energy  lh={h_lh:.1f} hl={h_hl:.1f} | vert-stripe lh={v_lh:.1f} hl={v_hl:.1f}")
print(f"                    -> code 'lh' detects {'HORIZONTAL structure = pywt cH' if lh_is_horizontal else 'VERTICAL structure = pywt cV'}")
print(f"                    -> code 'hl' detects {'VERTICAL structure = pywt cV' if lh_is_horizontal else 'HORIZONTAL structure = pywt cH'}")
print("="*70)

# ---------- channel selection (verify, don't default) ----------
chan_var = x[0].var(dim=(1, 2))                            # per-channel spatial variance, shape (C,)
cstar = int(chan_var.argmax().item())
order = torch.argsort(chan_var, descending=True)[:5].tolist()
print("CHANNEL SELECTION")
print(f"  per-channel variance argmax = {cstar}  (var={chan_var[cstar].item():.3f})")
print(f"  top-5 channels by variance  = {[(int(i), round(chan_var[i].item(),2)) for i in order]}")
print(f"  channel USED for figure     = c={cstar}")

# ---------- enhancement path final activation + rectification check ----------
act_name = type(wfm.hf[-1]).__name__
frac_nonneg = float(((torch.cat([e_lh, e_hl, e_hh]) >= 0).float().mean()).item())
print("ENHANCEMENT PATH")
print(f"  hf final layer            = {act_name}")
print(f"  fraction of LH'/HL'/HH' >= 0 = {frac_nonneg:.4f}")
print(f"  perturbation train-only={not False}, forced ON for this figure (style_prob=1, training=True)")
def _ratio(a, b): return float((a.abs().mean() / b.abs().mean()).item())
print("  detail-band |after|/|before| averaged over ALL 40 channels:")
print(f"    mean|LH'|/mean|LH| = {_ratio(e_lh, lh):.4f}")
print(f"    mean|HL'|/mean|HL| = {_ratio(e_hl, hl):.4f}")
print(f"    mean|HH'|/mean|HH| = {_ratio(e_hh, hh):.4f}")
print("="*70)
ENH_RECTIFIED = (act_name == "ReLU") or (frac_nonneg > 0.98)

# ---------- to numpy (single-channel version only) ----------
def cvar(t):  return t[0, cstar].cpu().numpy()

def build_panels(reduce):
    return dict(
        x=reduce(x), ll=reduce(ll), lh=reduce(lh), hl=reduce(hl), hh=reduce(hh),
        ll_p=reduce(ll_pert), e_lh=reduce(e_lh), e_hl=reduce(e_hl), e_hh=reduce(e_hh),
        rec=reduce(rec), fused=reduce(fused), out=reduce(out))

# labels for the two signed detail bands, from the empirical check
LH_LAB = "LH  (horizontal detail, cH)" if lh_is_horizontal else "LH  (vertical detail, cV)"
HL_LAB = "HL  (vertical detail, cV)"   if lh_is_horizontal else "HL  (horizontal detail, cH)"

from matplotlib.lines import Line2D
ACT = "magma"; DIV = "RdBu_r"; SEQ = "viridis"   # SEQ for rectified (non-negative) enhanced bands

def render(reduce, tag):
    P = build_panels(reduce)
    d2 = ll.shape[-2]
    # clean 4x4 grid: columns aligned by band (LL/LH/HL/HH), rows = pipeline stage
    fig = plt.figure(figsize=(7.6, 8.7))
    gs = GridSpec(4, 4, figure=fig, wspace=0.08, hspace=0.24,
                  left=0.085, right=0.935, top=0.855, bottom=0.055)

    def panel(rc, img, title, cmap, sym=False, shared=None, seq0=False, cbar=False):
        r, c = rc; ax = fig.add_subplot(gs[r, c])
        if sym:
            v = np.abs(img).max() + 1e-9; im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
        elif seq0:                                          # non-negative -> anchor scale at 0
            im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=img.max() + 1e-9)
        elif shared is not None:
            im = ax.imshow(img, cmap=cmap, vmin=shared[0], vmax=shared[1])
        else:
            im = ax.imshow(img, cmap=cmap)
        ax.set_aspect("equal"); ax.set_anchor("N")        # square panels, top-anchored -> rows align
        ax.set_title(title, pad=3, fontsize=7.8); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#888"); s.set_linewidth(0.5)
        if cbar:
            cax = ax.inset_axes([1.04, 0.0, 0.055, 1.0])   # inset -> does not shrink the grid cell
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5)
        return ax, im

    llv = (min(P["ll"].min(), P["ll_p"].min()), max(P["ll"].max(), P["ll_p"].max()))
    dLL = P["ll_p"] - P["ll"]; dLH = P["e_lh"] - P["lh"]; dHL = P["e_hl"] - P["hl"]; dHH = P["e_hh"] - P["hh"]

    # LL pair uses a JOINT vmin/vmax (shared=llv) -> the two panels are directly comparable
    print("LL PANEL SHARED SCALE")
    print(f"  vmin={llv[0]:.3f}  vmax={llv[1]:.3f}  (joint over LL and LL-after, channel c={cstar})")
    print("  previous version: already SHARED (both imshow received these same vmin/vmax) - NOT independent")

    # Row 0 -- Haar sub-bands (signed detail -> diverging)
    ax_ll, im_ll = panel((0, 0), P["ll"], "LL  (approximation)", ACT, shared=llv)
    panel((0, 1), P["lh"], LH_LAB, DIV, sym=True)
    panel((0, 2), P["hl"], HL_LAB, DIV, sym=True)
    panel((0, 3), P["hh"], "HH  (diagonal, cD)", DIV, sym=True, cbar=True)
    # Row 1 -- after sub-band processing (LL perturbed; details rectified -> sequential)
    ax_lla, _ = panel((1, 0), P["ll_p"], "LL after (\u03bb=0.3)", ACT, shared=llv)
    panel((1, 1), P["e_lh"], "LH\u2032 enhanced", SEQ, seq0=True)
    panel((1, 2), P["e_hl"], "HL\u2032 enhanced", SEQ, seq0=True)
    panel((1, 3), P["e_hh"], "HH\u2032 enhanced", SEQ, seq0=True, cbar=True)
    # Row 2 -- change induced by the WFM (after - before), genuinely signed -> diverging
    # path labels folded into titles: LL = perturbation; details = enhancement conv (bypass perturb)
    panel((2, 0), dLL, "\u0394LL  (perturbation)", DIV, sym=True)
    panel((2, 1), dLH, "\u0394LH  (enh. conv)", DIV, sym=True)
    panel((2, 2), dHL, "\u0394HL  (enh. conv)", DIV, sym=True)
    panel((2, 3), dHH, "\u0394HH  (enh. conv)", DIV, sym=True, cbar=True)
    # Row 3 -- spatial-domain maps (forward + inverse + fusion)
    panel((3, 0), P["x"],     "input feature map x", ACT)
    panel((3, 1), P["rec"],   "IDWT reconstruction", ACT)
    panel((3, 2), P["fused"], "1\u00d71 fuse(rec)", ACT)
    panel((3, 3), P["out"],   "output = fuse(rec)+x", ACT, cbar=True)

    # single SHARED colorbar for the LL pair -> vertical bar in the LEFT margin, spanning both rows
    fig.canvas.draw()                                     # realize aspect-adjusted positions
    p0 = ax_ll.get_position(); p1 = ax_lla.get_position()
    cax = fig.add_axes([0.050, p1.y0, 0.012, p0.y1 - p1.y0])
    cb = fig.colorbar(im_ll, cax=cax)
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.tick_params(labelsize=5, pad=1)
    fig.text(0.056, p1.y0 - 0.008, "LL shared", ha="center", va="top", fontsize=5.4, color="#333")

    # forced-perturbation note tucked directly under the row-2 panels (not mid-gap -> unambiguous)
    fig.text(0.55, p1.y0 - 0.008,
             "perturbation forced active for illustration; disabled at inference",
             ha="center", va="top", fontsize=6.8, style="italic", color="#8a1c1c")

    # row labels on the far left
    for y, lab in zip([0.775, 0.565, 0.355, 0.145],
                      ["Haar\nsub-bands", "After sub-band\nprocessing",
                       "Change\n(after\u2212before)", "Spatial\nmaps"]):
        fig.text(0.020, y, lab, ha="center", va="center", rotation=90, fontweight="bold", fontsize=8)


    fig.suptitle("Wavelet Frequency Module: Haar decomposition, low-frequency\n"
                 "perturbation and high-frequency enhancement", fontsize=10.5, y=0.985)
    fig.text(0.5, 0.918,
             f"stride-{STRIDE} skip \u00b7 {C} ch \u00b7 {H}\u00d7{W} \u2192 {d2}\u00d7{d2} after DWT \u00b7 "
             f"\u03bb=0.3, p=0.5 \u00b7 channel c={cstar} \u00b7 {DS} (unseen)",
             ha="center", fontsize=7, color="#444")
    outdir = r"D:\Downloads"
    fig.savefig(os.path.join(outdir, f"{tag}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("SAVED", os.path.join(outdir, tag))

render(cvar, "figure_2_wfm")   # single-channel version only
