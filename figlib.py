# Shared infrastructure for genuine paper figures: model + data + style + helpers.
# Every figure built on this uses the REAL trained checkpoint and REAL datasets.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(os.getcwd(), "torch_cache"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF

# ---- palette (elsevier-tikz RYB cycle; red/green reserved for directional meaning) ----
INK, INK2, MUTED = "#111111", "#454545", "#8a8a8a"
BLUE   = "#255BCF"   # RYB2 - primary signal
RED    = "#CF2525"   # RYB1 - directional only (loss / FP / decline)
GREEN  = "#25CF5B"   # RYB3 - directional only (gain / TP)
MAG    = "#A31A91"   # RYB4
ORNG   = "#FDB462"   # RYB5 - accent (e.g. 'unseen')
LGREEN = "#B3DE69"   # RYB6
LBLUE  = "#80B1D3"   # RYB7
GRID, BASE = "#e6e6e6", "#b9b9b9"
# semantic roles
SEEN, UNSEEN = BLUE, "#E8892B"      # warm accent for unseen (not red/green)
SEEN_L, UNSEEN_L = LBLUE, "#F6C79A"
GREEN_D, RED_D = "#1a9e46", "#b31f1f"

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
        "font.size": 8, "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222", "axes.labelcolor": INK, "text.color": INK,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "xtick.color": "#222222", "ytick.color": "#222222",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        # full-box axes (SCI convention)
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })
set_style()
def full_box(ax):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.8)
    ax.tick_params(direction="out", top=False, right=False, length=3, width=0.7)
FIGD = Path("outputs/figures"); FIGD.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    import time as _t
    def _try(path, **kw):
        try:
            fig.savefig(path, **kw); return path
        except PermissionError:
            alt = str(path).replace(name, f"{name}_{_t.strftime('%H%M%S')}")
            fig.savefig(alt, **kw); print("  (locked, wrote", alt, ")"); return alt
    _try(FIGD/f"{name}.pdf", bbox_inches="tight")
    p = _try(FIGD/f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig); print("SAVED", p)

# ---- data ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG = 352
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
IMG_EXT = (".png",".jpg",".jpeg",".tif",".tiff",".bmp")
TEST_SETS = {
 "Kvasir-SEG":       ("seen",  "data/Kvasir-SEG/images","data/Kvasir-SEG/masks"),
 "CVC-ClinicDB":     ("seen",  "data/_kaggle/cvcclinicdb/PNG/Original","data/_kaggle/cvcclinicdb/PNG/Ground Truth"),
 "CVC-ColonDB":      ("unseen","data/_kaggle/cvc-colondb/CVC-ColonDB/images","data/_kaggle/cvc-colondb/CVC-ColonDB/masks"),
 "ETIS-LaribPolypDB":("unseen","data/_kaggle/etis-larib/images","data/_kaggle/etis-larib/masks"),
 "CVC-300":          ("unseen","data/_kaggle/cvc-300/CVC-300/images","data/_kaggle/cvc-300/CVC-300/masks"),
}
SHORT = {"Kvasir-SEG":"Kvasir-SEG","CVC-ClinicDB":"CVC-ClinicDB","CVC-ColonDB":"CVC-ColonDB",
         "ETIS-LaribPolypDB":"ETIS-Larib","CVC-300":"CVC-300"}
# REAL verified metrics from the 100ep/352/TTA run (best checkpoint)
REPORT = {
 "Kvasir-SEG":("seen",0.9673,0.9394,0.9712,0.9935,0.9663),
 "CVC-ClinicDB":("seen",0.9635,0.9315,0.9667,0.9966,0.9622),
 "CVC-ColonDB":("unseen",0.7395,0.6598,0.7602,0.9912,0.7687),
 "ETIS-LaribPolypDB":("unseen",0.7001,0.6227,0.7834,0.9797,0.6764),
 "CVC-300":("unseen",0.8867,0.8149,0.9639,0.9938,0.8423),
}

def list_pairs(img_dir, mask_dir):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    mby = {p.stem:p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = [(im, mby[im.stem]) for im in imgs if im.stem in mby]
    if not pairs:
        masks = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT]); pairs = list(zip(imgs, masks))
    return pairs
def load_item(ip, mp, size=IMG):
    img = Image.open(ip).convert("RGB").resize((size,size), Image.BILINEAR)
    msk = Image.open(mp).convert("L").resize((size,size), Image.NEAREST)
    return TF.normalize(TF.to_tensor(img), MEAN, STD), (TF.to_tensor(msk) > 0.5).float()
def denorm(t):
    t = t.detach().cpu().float(); m = torch.tensor(MEAN).view(3,1,1); s = torch.tensor(STD).view(3,1,1)
    return torch.clamp(t*s+m, 0, 1).permute(1,2,0).numpy()
def dice_of(pr, gt, thr=0.5):
    p = (pr > thr).astype(np.float32); t = (gt > 0.5).astype(np.float32)
    return float((2*(p*t).sum()+1)/(p.sum()+t.sum()+1))

# ---- model (verbatim from run.py) ----
def _haar(device, dtype):
    h=0.5
    return torch.stack([torch.tensor([[h,h],[h,h]],device=device,dtype=dtype),
        torch.tensor([[h,h],[-h,-h]],device=device,dtype=dtype),
        torch.tensor([[h,-h],[h,-h]],device=device,dtype=dtype),
        torch.tensor([[h,-h],[-h,h]],device=device,dtype=dtype)]).unsqueeze(1)
def dwt2(x):
    B,C,H,W=x.shape; f=_haar(x.device,x.dtype).repeat(C,1,1,1)
    if H%2: x=F.pad(x,(0,0,0,1))
    if W%2: x=F.pad(x,(0,1,0,0))
    o=F.conv2d(x,f,stride=2,groups=C).view(B,C,4,-1,x.shape[-1]//2 if W%2==0 else (x.shape[-1])//2)
    o=F.conv2d(x,f,stride=2,groups=C); o=o.view(B,C,4,o.shape[-2],o.shape[-1])
    return o[:,:,0],o[:,:,1],o[:,:,2],o[:,:,3]
def idwt2(ll,lh,hl,hh):
    B,C,h,w=ll.shape; f=_haar(ll.device,ll.dtype).repeat(C,1,1,1)
    return F.conv_transpose2d(torch.stack([ll,lh,hl,hh],dim=2).view(B,4*C,h,w),f,stride=2,groups=C)
class WFM(nn.Module):
    def __init__(self,ch,style_prob=0.5):
        super().__init__(); self.style_prob=style_prob
        self.hf=nn.Sequential(nn.Conv2d(3*ch,3*ch,3,padding=1,groups=ch),nn.BatchNorm2d(3*ch),nn.ReLU(inplace=True))
        self.fuse=nn.Conv2d(ch,ch,1)
    def _p(self,ll):
        if not self.training: return ll
        B=ll.shape[0]; do=(torch.rand(B,device=ll.device)<self.style_prob).float().view(B,1,1,1)
        mu=ll.mean(dim=(2,3),keepdim=True); sd=ll.std(dim=(2,3),keepdim=True)+1e-5
        g=1+0.3*torch.randn(B,1,1,1,device=ll.device); b=0.3*torch.randn(B,1,1,1,device=ll.device)
        return do*(((ll-mu)/sd)*(sd*g)+(mu+b*sd))+(1-do)*ll
    def forward(self,x):
        ll,lh,hl,hh=dwt2(x); ll=self._p(ll); B,C=x.shape[:2]
        hf=self.hf(torch.cat([lh,hl,hh],dim=1)).view(B,C,3,*lh.shape[-2:])
        rec=idwt2(ll,hf[:,:,0],hf[:,:,1],hf[:,:,2])
        rec=F.interpolate(rec,size=x.shape[-2:],mode="bilinear",align_corners=False); return self.fuse(rec)+x
class MambaLite(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.norm=nn.GroupNorm(8,ch); self.inp=nn.Conv2d(ch,2*ch,1)
        self.dw=nn.Conv2d(ch,ch,3,padding=1,groups=ch); self.a=nn.Parameter(torch.tensor(0.9)); self.out=nn.Conv2d(ch,ch,1)
    def _scan(self,x):
        B,C,H,W=x.shape; a=torch.sigmoid(self.a); seq=x.flatten(2).transpose(1,2)
        outs=torch.empty_like(seq); h=torch.zeros(B,C,device=x.device)
        for t in range(seq.shape[1]): h=a*h+(1-a)*seq[:,t]; outs[:,t]=h
        return outs.transpose(1,2).view(B,C,H,W)
    def forward(self,x):
        r=x; x=self.norm(x); u,g=self.inp(x).chunk(2,dim=1); u=self.dw(u); u=self._scan(u)*torch.sigmoid(g); return r+self.out(u)
class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__(); self.b=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU(inplace=True),
            nn.Conv2d(co,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU(inplace=True))
    def forward(self,x): return self.b(x)
class WaveMambaDG(nn.Module):
    def __init__(self,style_prob=0.5,pretrained=False,use_wfm=True,use_ssm=True):
        super().__init__(); import timm
        self.use_wfm=use_wfm
        self.enc=timm.create_model("mobilenetv3_large_100",features_only=True,pretrained=pretrained)
        chs=self.enc.feature_info.channels()
        self.wfm=nn.ModuleList([WFM(c,style_prob) for c in chs]) if use_wfm else None
        self.bottleneck=MambaLite(chs[-1]) if use_ssm else nn.Identity()
        rev=chs[::-1]; self.ups,self.dec=nn.ModuleList(),nn.ModuleList()
        for i in range(len(rev)-1):
            self.ups.append(nn.ConvTranspose2d(rev[i],rev[i+1],2,stride=2)); self.dec.append(ConvBlock(rev[i+1]*2,rev[i+1]))
        self.mask_head=nn.Conv2d(chs[0],1,1); self.bnd_head=nn.Conv2d(chs[0],1,1)
    def forward(self,x):
        feats=self.enc(x)
        if self.use_wfm: feats=[w(f) for w,f in zip(self.wfm,feats)]
        d=self.bottleneck(feats[-1]); skips=feats[::-1]
        for i,(up,dec) in enumerate(zip(self.ups,self.dec)):
            d=up(d); s=skips[i+1]
            if d.shape[-2:]!=s.shape[-2:]: d=F.interpolate(d,size=s.shape[-2:],mode="bilinear",align_corners=False)
            d=dec(torch.cat([d,s],dim=1))
        m=F.interpolate(self.mask_head(d),size=x.shape[-2:],mode="bilinear",align_corners=False)
        b=F.interpolate(self.bnd_head(d),size=x.shape[-2:],mode="bilinear",align_corners=False)
        return {"mask_logit":m,"boundary_logit":b}

_MODEL=None
def get_model():
    global _MODEL
    if _MODEL is None:
        print("loading checkpoint ...")
        _MODEL=WaveMambaDG(pretrained=False).to(DEVICE).eval()
        _MODEL.load_state_dict(torch.load("outputs/wavemamba_dg_best.pt",map_location=DEVICE,weights_only=True))
    return _MODEL
@torch.no_grad()
def predict(it):
    return torch.sigmoid(get_model()(it.unsqueeze(0).to(DEVICE))["mask_logit"])[0,0].cpu().numpy()
