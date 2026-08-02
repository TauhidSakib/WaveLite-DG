# Graphical abstract for WaveMamba-DG — Elsevier spec (531 x 1328 h x w, readable at 5 x 13 cm).
# Layout: title band -> full-width architecture pipeline -> 3 panels (Challenge / WFM / Results).
# Designed at final print size (cm). Palette: validated CVD-safe dataviz defaults.
import os
os.chdir(r"D:\AI-Projects\WaveMamba")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, BLUE_D, BLUE_T = "#2a78d6", "#184f95", "#e9f1fb"
ORNG, ORNG_D, ORNG_T = "#eb6834", "#b8461f", "#fdecdf"
GRID, BASE, GREEN = "#e1e0d9", "#c3c2b7", "#008300"
SURF, TISSUE, POLYP = "#ffffff", "#e7b6a3", "#b5654c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

W, H = 13.28, 5.31
fig = plt.figure(figsize=(W/2.54, H/2.54))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor(SURF)
ax.add_patch(Rectangle((0, 0), W, H, facecolor=SURF, edgecolor="none", zorder=-10))

def rbox(x, y, w, h, fc, ec, lw=0.8, z=3, rs=0.09, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha))

def shadow(x, y, w, h, off=0.045, rs=0.09, z=2):
    ax.add_patch(FancyBboxPatch((x+off, y-off), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                 linewidth=0, facecolor="#000000", alpha=0.055, zorder=z))

def arrow(x0, y0, x1, y1, color=INK2, lw=1.0, z=6, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))

def txt(x, y, s, size, color=INK, weight="normal", ha="center", va="center", z=9, style="normal", sp=1.0):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                zorder=z, style=style); t.set_linespacing(sp); return t

def novel_tag(cx, y):
    rbox(cx-0.36, y, 0.72, 0.24, ORNG, "none", rs=0.12, z=8)
    txt(cx, y+0.12, "NOVEL", 5.0, "white", "bold", z=9)

# ===== TITLE =====
txt(0.34, 4.92, "WaveLite‑DG", 15, INK, "bold", ha="left")
txt(0.36, 4.52,
    "A Lightweight Wavelet‑Guided Hybrid Network for Generalizable, Real‑Time Polyp Segmentation",
    7.0, INK2, ha="left")
ax.plot([0.35, 12.94], [4.30, 4.30], color=BLUE, lw=1.6, zorder=4, solid_capstyle="round")
ax.plot([0.35, 3.05], [4.30, 4.30], color=ORNG, lw=1.6, zorder=5, solid_capstyle="round")

# ===== ARCHITECTURE PIPELINE (full width) =====
txt(0.36, 4.06, "ARCHITECTURE", 6.4, INK2, "bold", ha="left")
centers = [1.28, 3.44, 5.60, 7.76, 9.92, 12.08]
half, by, bh = 0.775, 3.02, 0.80
mid = by + bh/2

def stage(cx, title, sub, novel=False):
    rbox(cx-half, by, 2*half, bh, ORNG_T if novel else BLUE_T, ORNG if novel else BLUE, 0.9, z=4)
    ty = by+bh-0.24 if sub else mid
    txt(cx, ty, title, 6.2, INK, "bold", sp=0.95)
    if sub: txt(cx, by+0.18, sub, 4.8, INK2)
    if novel: novel_tag(cx, by+bh+0.02)

# input frame
rbox(centers[0]-half, by, 2*half, bh, TISSUE, BLUE, 0.9, z=4)
ax.add_patch(Ellipse((centers[0], mid+0.03), 0.42, 0.30, facecolor=POLYP, edgecolor="none", zorder=5))
txt(centers[0], by-0.17, "colonoscopy frame", 4.9, INK2)
arrow(centers[0]+half, mid, centers[1]-half, mid)
stage(centers[1], "Encoder", "MobileNetV3‑lite")
arrow(centers[1]+half, mid, centers[2]-half, mid)
stage(centers[2], "Wavelet Freq.\nModule", None, novel=True)
arrow(centers[2]+half, mid, centers[3]-half, mid)
stage(centers[3], "Gated context", "linear‑time")
arrow(centers[3]+half, mid, centers[4]-half, mid)
stage(centers[4], "Boundary‑aware\nDecoder", None)
arrow(centers[4]+half, mid, centers[5]-half, mid)
# output frame
rbox(centers[5]-half, by, 2*half, bh, TISSUE, GREEN, 1.0, z=4)
ax.add_patch(Ellipse((centers[5], mid+0.03), 0.42, 0.30, facecolor=POLYP, edgecolor="none", zorder=5))
ax.add_patch(Ellipse((centers[5], mid+0.03), 0.50, 0.38, facecolor="none", edgecolor=GREEN, lw=1.2, zorder=6))
txt(centers[5], by-0.17, "segmentation mask", 4.9, GREEN, "bold")

# dividers for lower panels
for xd in (4.34, 8.72):
    ax.plot([xd, xd], [0.34, 2.62], color=GRID, lw=0.7, zorder=1)

# ===== PANEL 1 · CHALLENGE =====
txt(0.36, 2.52, "CLINICAL CHALLENGE", 6.3, BLUE_D, "bold", ha="left")
txt(0.40, 2.22, "TRAIN · 2 source centers", 5.6, BLUE_D, "bold", ha="left")
rbox(0.40, 1.78, 1.72, 0.40, BLUE_T, BLUE, 0.8, z=3); txt(1.26, 1.98, "Kvasir‑SEG", 5.7, INK, "bold")
rbox(2.22, 1.78, 1.90, 0.40, BLUE_T, BLUE, 0.8, z=3); txt(3.17, 1.98, "CVC‑ClinicDB", 5.7, INK, "bold")
arrow(0.95, 1.74, 0.95, 1.40, ORNG, 1.2, ms=8)
txt(1.14, 1.57, "domain shift", 5.4, ORNG_D, "bold", ha="left", style="italic")
txt(0.40, 1.24, "DEPLOY · 3 unseen centers", 5.6, ORNG_D, "bold", ha="left")
for i, nm in enumerate(["ColonDB", "ETIS", "CVC‑300"]):
    x0 = 0.40 + i*1.26
    rbox(x0, 0.82, 1.14, 0.38, ORNG_T, ORNG, 0.8, z=3); txt(x0+0.57, 1.01, nm, 5.4, INK, "bold")
txt(0.40, 0.52, "Scanner & population shift degrade", 5.4, INK2, ha="left")
txt(0.40, 0.32, "accuracy at unseen hospitals.", 5.4, INK2, ha="left")

# ===== PANEL 2 · WAVELET FREQUENCY MODULE =====
txt(4.46, 2.52, "WAVELET FREQUENCY MODULE", 6.3, ORNG_D, "bold", ha="left")
ey, eh = 1.66, 0.44
rbox(4.46, ey, 0.72, eh, "#eef2f6", MUTED, 0.8, z=3); txt(4.82, ey+eh/2, "DWT", 5.2, INK, "bold")
arrow(5.18, ey+eh/2, 5.42, ey+eh/2, INK2, 0.8, ms=5)
rbox(5.44, ey+0.28, 2.05, 0.36, ORNG_T, ORNG, 0.8, z=3)
txt(6.46, ey+0.46, "LL low‑freq → style perturb", 4.6, INK, "bold")
rbox(5.44, ey-0.14, 2.05, 0.36, BLUE_T, BLUE, 0.8, z=3)
txt(6.46, ey+0.04, "LH·HL·HH → boundary", 4.6, INK, "bold")
arrow(7.51, ey+0.46, 7.74, ey+eh/2, ORNG, 0.7, ms=5)
arrow(7.51, ey+0.04, 7.74, ey+eh/2, BLUE, 0.7, ms=5)
rbox(7.76, ey, 0.80, eh, "#eef2f6", MUTED, 0.8, z=3); txt(8.16, ey+eh/2, "IDWT", 5.2, INK, "bold")
txt(4.46, 1.14, "Style lives in low frequencies; anatomy in high —", 5.1, INK2, ha="left", style="italic")
txt(4.46, 0.92, "perturb one, preserve the other → domain‑invariant.", 5.1, INK2, ha="left", style="italic")
txt(4.46, 0.56, "+ boundary‑aware decoder & gated context block", 5.0, MUTED, ha="left")
txt(4.46, 0.36, "  global context at low parameter cost.", 5.0, MUTED, ha="left")

# ===== PANEL 3 · RESULTS =====
txt(8.84, 2.52, "RESULTS", 6.3, BLUE_D, "bold", ha="left")
# legend
rbox(10.05, 2.44, 0.15, 0.15, BLUE, "none", rs=0.05, z=6); txt(10.27, 2.515, "Seen", 5.0, INK2, ha="left")
rbox(11.05, 2.44, 0.15, 0.15, ORNG, "none", rs=0.05, z=6); txt(11.27, 2.515, "Unseen", 5.0, INK2, ha="left")

axb = fig.add_axes([9.00/W, 1.50/H, (12.86-9.00)/W, (2.34-1.50)/H]); axb.set_facecolor("none")
labels = ["Kvasir", "ClinicDB", "ColonDB", "ETIS", "CVC‑300"]
vals = [0.9673, 0.9635, 0.7395, 0.7001, 0.8867]
cols = [BLUE, BLUE, ORNG, ORNG, ORNG]
xp = np.arange(5)
axb.bar(xp, vals, width=0.72, color=cols, zorder=3, edgecolor="white", linewidth=0.5)
for xi, v in zip(xp, vals):
    axb.text(xi, v+0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4, fontweight="bold", color=INK)
axb.set_ylim(0, 1.15); axb.set_xlim(-0.6, 4.6)
axb.set_xticks(xp); axb.set_xticklabels(labels, fontsize=4.9, color=INK2, rotation=20, ha="right")
axb.set_yticks([0, 0.5, 1.0]); axb.set_yticklabels(["0", ".5", "1"], fontsize=4.6, color=MUTED)
axb.tick_params(length=0, pad=1.2)
for s in ["top", "right"]:
    axb.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    axb.spines[s].set_color(BASE); axb.spines[s].set_linewidth(0.6)
axb.axhline(0.5, color=GRID, lw=0.5, zorder=1); axb.axhline(1.0, color=GRID, lw=0.5, zorder=1)

# KPI tiles
tiles = [("7.65 M", "parameters"), ("1.93 G", "FLOPs (lowest)"), ("2 → 3", "train → unseen")]
tw, gap, tx = 1.20, 0.13, 8.84
for num, sub in tiles:
    shadow(tx, 0.28, tw, 0.66)
    rbox(tx, 0.28, tw, 0.66, "#f6faff", BLUE, 0.8, z=4)
    txt(tx+tw/2, 0.72, num, 9.3, BLUE_D, "bold")
    txt(tx+tw/2, 0.43, sub, 4.6, INK2)
    tx += tw + gap

out = "outputs"
fig.savefig(f"{out}/graphical_abstract.pdf", facecolor=SURF)
fig.savefig(f"{out}/graphical_abstract.tiff", dpi=600, facecolor=SURF, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{out}/graphical_abstract.png", dpi=600, facecolor=SURF)
from PIL import Image
print("saved. TIFF (w x h):", Image.open(f"{out}/graphical_abstract.tiff").size)
