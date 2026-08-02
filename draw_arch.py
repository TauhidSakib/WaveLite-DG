# Simple boxes+arrows architecture sketch of WaveLite-DG -> PNG for redrawing in draw.io.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 100); ax.set_ylim(-30, 100); ax.axis("off")

C = dict(enc="#cfe3f7", wfm="#cfeecd", mamba="#f7dcc0", dec="#e2e2e2",
         head="#f6c9c9", io="#efe6fb")
EC = dict(enc="#2b6cb0", wfm="#2f8f2f", mamba="#c9761f", dec="#666666",
          head="#c0392b", io="#7a4fbf")

def box(x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                 linewidth=1.6, edgecolor=ec, facecolor=fc, mutation_scale=1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111111")
    return (x, y, w, h)

def arrow(p1, p2, style="-|>", color="#333333", lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                 lw=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}"))

def c_bottom(b): x,y,w,h=b; return (x+w/2, y)
def c_top(b):    x,y,w,h=b; return (x+w/2, y+h)
def c_left(b):   x,y,w,h=b; return (x, y+h/2)
def c_right(b):  x,y,w,h=b; return (x+w, y+h/2)

# ---- column x positions ----
xe, xw, xd = 6, 26, 66          # encoder / WFM / decoder columns
BW, BH = 15, 6.5

# vertical levels
lv = dict(inp=90, s0=78, s1=66, s2=54, s3=42, s4=30, bott=16)

# ---------- encoder chain ----------
inp = box(xe, lv["inp"], BW, BH, "Input\n3 x 352 x 352", C["io"], EC["io"], bold=True)
enc = {}
enc_spec = [("s0","Enc S0\n16 x 176 x 176"),("s1","Enc S1\n24 x 88 x 88"),
            ("s2","Enc S2\n40 x 44 x 44"),("s3","Enc S3\n112 x 22 x 22"),
            ("s4","Enc S4\n960 x 11 x 11")]
for k,t in enc_spec:
    enc[k] = box(xe, lv[k], BW, BH, t, C["enc"], EC["enc"])
ax.text(xe+BW/2, lv["inp"]+BH+3.5, "MobileNetV3-Large encoder", ha="center",
        fontsize=10, fontweight="bold", color=EC["enc"])

# encoder downward arrows
arrow(c_bottom(inp), c_top(enc["s0"]))
for a,b in [("s0","s1"),("s1","s2"),("s2","s3"),("s3","s4")]:
    arrow(c_bottom(enc[a]), c_top(enc[b]))

# ---------- WFM per stage ----------
wfm = {}
for k in ["s0","s1","s2","s3","s4"]:
    wfm[k] = box(xw, lv[k], BW, BH, "WFM\n(wavelet freq)", C["wfm"], EC["wfm"])
    arrow(c_right(enc[k]), c_left(wfm[k]))
ax.text(xw+BW/2, lv["s0"]+BH+2.2, "Wavelet-Freq Module (x5)", ha="center",
        fontsize=9.5, fontweight="bold", color=EC["wfm"])

# ---------- bottleneck MambaLite ----------
mob = box((xw+xd)/2-2, lv["bott"]-1, BW+4, BH, "MambaLite block\n(gated linear-scan / SSM-style)\n960 x 11 x 11",
          C["mamba"], EC["mamba"], fs=8.5, bold=True)
arrow(c_bottom(wfm["s4"]), c_top(mob), rad=-0.15)

# ---------- decoder chain (right, going up) ----------
dec = {}
dec_spec = [("s3","Up0  960->112\nConvBlock -> 112 x 22 x 22"),
            ("s2","Up1  112->40\nConvBlock -> 40 x 44 x 44"),
            ("s1","Up2  40->24\nConvBlock -> 24 x 88 x 88"),
            ("s0","Up3  24->16\nConvBlock -> 16 x 176 x 176")]
for k,t in dec_spec:
    dec[k] = box(xd, lv[k], BW+6, BH, t, C["dec"], EC["dec"], fs=8)
ax.text(xd+(BW+6)/2, lv["inp"]+BH+3.5, "U-Net decoder (upsample + concat skip)", ha="center",
        fontsize=10, fontweight="bold", color="#444444")

# bottleneck -> up0, then upward chain
arrow(c_right(mob), c_bottom(dec["s3"]), rad=0.15)
for a,b in [("s3","s2"),("s2","s1"),("s1","s0")]:
    arrow(c_top(dec[a]), c_bottom(dec[b]))

# skip connections WFM -> decoder concat (dashed)
for k in ["s0","s1","s2","s3"]:
    arrow(c_right(wfm[k]), c_left(dec[k]), color=EC["wfm"], ls="--", lw=1.6, style="-|>")
ax.text((xw+xd)/2+4, lv["s0"]+2.4, "skip (concat)", ha="center", fontsize=8,
        color=EC["wfm"], style="italic")

# ---------- heads ----------
hm = box(xd, lv["inp"], (BW+6)/2-0.5, BH, "Mask head\n1x1 -> 352^2", C["head"], EC["head"], fs=8, bold=True)
hb = box(xd+(BW+6)/2+0.5, lv["inp"], (BW+6)/2-0.5, BH, "Boundary head\n1x1 -> 352^2", C["head"], EC["head"], fs=8, bold=True)
arrow(c_top(dec["s0"]), c_bottom(hm))
arrow(c_top(dec["s0"]), c_bottom(hb), rad=0.1)
ax.text(xd+(BW+6)/2, lv["inp"]+BH+1.6, "Dice+BCE (mask) + boundary loss", ha="center",
        fontsize=8, color=EC["head"], style="italic")

# ================= detail insets (below) =================
def small(x,y,w,h,t,fc,ec,fs=7.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1,rounding_size=0.8",
                 linewidth=1.3,edgecolor=ec,facecolor=fc))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal")
    return (x,y,w,h)

# WFM detail
y0=-26
ax.text(3, y0+18, "WFM detail", fontsize=10, fontweight="bold", color=EC["wfm"])
a=small(2,y0+8,10,5,"DWT (Haar)",C["wfm"],EC["wfm"])
b1=small(15,y0+12,13,4.5,"LL -> Style Perturb\n(train only)",C["wfm"],EC["wfm"])
b2=small(15,y0+4,15,4.5,"LH,HL,HH -> DWConv3x3\n+BN+ReLU",C["wfm"],EC["wfm"])
c=small(33,y0+8,9,5,"IDWT\nresize",C["wfm"],EC["wfm"])
d=small(45,y0+8,10,5,"Conv1x1\n(+) residual",C["wfm"],EC["wfm"],bold=True)
arrow(c_right(a),c_left(b1),color=EC["wfm"]); arrow(c_right(a),c_left(b2),color=EC["wfm"])
arrow(c_right(b1),c_left(c),color=EC["wfm"]); arrow(c_right(b2),(c[0],c[1]+c[3]/2),color=EC["wfm"])
arrow(c_right(c),c_left(d),color=EC["wfm"])

# MambaLite detail
ax.text(60, y0+18, "MambaLite detail", fontsize=10, fontweight="bold", color=EC["mamba"])
m1=small(59,y0+8,11,5,"GroupNorm\nConv1x1 x2ch",C["mamba"],EC["mamba"])
m2=small(72,y0+8,10,5,"split u,g\nDWConv3x3(u)",C["mamba"],EC["mamba"])
m3=small(84,y0+8,7,5,"Linear\nScan",C["mamba"],EC["mamba"])
m4=small(59+6,y0+0.5,11,4,"* sigmoid(g)  ->  Conv1x1  ->  (+) residual",C["mamba"],EC["mamba"],fs=6.8,bold=True)
arrow(c_right(m1),c_left(m2),color=EC["mamba"]); arrow(c_right(m2),c_left(m3),color=EC["mamba"])
arrow(c_bottom(m3),(m4[0]+m4[2]*0.8,m4[1]+m4[3]),color=EC["mamba"],rad=-0.2)

ax.set_title("WaveLite-DG  —  architecture sketch (boxes & connections)",
             fontsize=14, fontweight="bold", pad=12)

out = r"C:\Users\Asus\Downloads\WaveLite_DG_architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("SAVED", out)
