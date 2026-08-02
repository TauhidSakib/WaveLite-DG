# Auto-generated from WaveMamba_DG_end2end.ipynb for headless CLI execution.
import os
ROOT = r"D:\AI-Projects\WaveMamba"
os.environ.setdefault("TMP", os.path.join(ROOT, "tmp"))
os.environ.setdefault("TEMP", os.path.join(ROOT, "tmp"))
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(ROOT, "kagglehub_cache"))
os.environ.setdefault("KAGGLE_CONFIG_DIR", r"C:\Users\Asus\.kaggle")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "torch_cache"))
os.chdir(ROOT)
import matplotlib
matplotlib.use("Agg")
print(">>> run.py starting in", os.getcwd(), flush=True)


# ==== cell ====
# If any import fails, uncomment the pip line, run once, then restart the kernel.
# %pip install torch torchvision timm pywavelets kagglehub scikit-learn matplotlib seaborn tqdm pillow requests

import torch, platform
print("Python :", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== cell ====
from pathlib import Path

class CFG:
    SEED        = 42
    IMG_SIZE    = int(os.environ.get("WM_IMG", 224))
    BATCH_SIZE  = int(os.environ.get("WM_BATCH", 8))
    EPOCHS      = int(os.environ.get("WM_EPOCHS", 15))   # demo value — use 80–120 for real results
    LR          = 1e-4
    WEIGHT_DECAY= 1e-4
    VAL_SPLIT   = 0.2
    NUM_WORKERS = 0           # set 0 on Windows if you hit loader errors
    WAVELET     = "haar"
    STYLE_PROB  = 0.5         # prob. of applying low-freq style perturbation (train only)
    ACCUM       = int(os.environ.get("WM_ACCUM", 1))     # gradient accumulation steps
    AMP         = os.environ.get("WM_AMP", "1") == "1"   # mixed precision (halves VRAM)
    TTA         = os.environ.get("WM_TTA", "0") == "1"   # test-time aug at evaluation
    MAXITERS    = int(os.environ.get("WM_MAXITERS", 0))  # 0 = full epoch; >0 = probe only
    DATA_DIR    = Path("data")
    OUT_DIR     = Path("outputs")
    DEVICE      = DEVICE

CFG.DATA_DIR.mkdir(exist_ok=True); CFG.OUT_DIR.mkdir(exist_ok=True)

import random, numpy as np
random.seed(CFG.SEED); np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
print("Config ready. Output dir:", CFG.OUT_DIR.resolve())


# ==== cell ====
import os, zipfile, io, shutil, requests
from pathlib import Path

def _find_pairs(root, img_hints=("images","image","original","png","pngimages"),
                mask_hints=("masks","mask","ground truth","groundtruth","gt")):
    # Locate an image folder and a mask folder anywhere under `root`.
    root = Path(root)
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    def score(p, hints):
        n = p.name.lower().replace("-"," ").replace("_"," ")
        return any(h in n for h in hints)
    img_dirs  = [p for p in dirs if score(p, img_hints)]
    mask_dirs = [p for p in dirs if score(p, mask_hints)]
    # prefer dirs that actually contain image files
    def has_imgs(p): return any(f.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff",".bmp") for f in p.iterdir() if f.is_file())
    img_dirs  = [p for p in img_dirs if has_imgs(p)] or img_dirs
    mask_dirs = [p for p in mask_dirs if has_imgs(p)] or mask_dirs
    return (img_dirs[0] if img_dirs else None, mask_dirs[0] if mask_dirs else None)

# ---- Kvasir-SEG : direct download ----
def get_kvasir_seg(dst):
    dst = Path(dst); img = dst/"Kvasir-SEG"/"images"; msk = dst/"Kvasir-SEG"/"masks"
    if img.exists() and msk.exists() and any(img.iterdir()):
        print("  Kvasir-SEG already present."); return img, msk
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    print("  downloading Kvasir-SEG ...")
    r = requests.get(url, timeout=120); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z: z.extractall(dst)
    img, msk = _find_pairs(dst/"Kvasir-SEG")
    print("     images:", img, "| masks:", msk)
    return img, msk

# ---- kagglehub datasets ----
KAGGLE_SLUGS = {
    "CVC-ClinicDB": "balraj98/cvcclinicdb",
    "PraNet-TestData": "debeshjha1/pranet-testdataset",  # bundles ColonDB/ETIS/CVC-300 (if available)
}

def get_kagglehub(slug):
    import kagglehub
    print(f"  kagglehub download: {slug}")
    path = kagglehub.dataset_download(slug)
    print("     ->", path)
    return Path(path)

DATA = {}
# Kvasir-SEG (always)
try:
    DATA["Kvasir-SEG"] = get_kvasir_seg(CFG.DATA_DIR)
except Exception as e:
    print("  !! Kvasir-SEG download failed:", e)

# CVC-ClinicDB + unseen test sets — pre-downloaded locally to data/_kaggle (explicit paths).
LOCAL_PAIRS = {
    "CVC-ClinicDB":      ("data/_kaggle/cvcclinicdb/PNG/Original",
                          "data/_kaggle/cvcclinicdb/PNG/Ground Truth"),
    "CVC-ColonDB":       ("data/_kaggle/cvc-colondb/CVC-ColonDB/images",
                          "data/_kaggle/cvc-colondb/CVC-ColonDB/masks"),
    "ETIS-LaribPolypDB": ("data/_kaggle/etis-larib/images",
                          "data/_kaggle/etis-larib/masks"),
    "CVC-300":           ("data/_kaggle/cvc-300/CVC-300/images",
                          "data/_kaggle/cvc-300/CVC-300/masks"),
}
for name, (im, mk) in LOCAL_PAIRS.items():
    im, mk = Path(im), Path(mk)
    if im.exists() and mk.exists():
        DATA[name] = (im, mk)
    else:
        print(f"  !! {name} local folder missing: {im} | {mk}")

print("\nResolved dataset folders:")
for k,v in DATA.items(): print(f"  {k:16s}: {v}")


# ==== cell ====
from pathlib import Path

def locate_test_subsets(bundle_root):
    out = {}
    if not bundle_root: return out
    bundle_root = Path(bundle_root)
    wanted = ["CVC-300","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","Kvasir"]
    for w in wanted:
        for cand in bundle_root.rglob(w):
            if cand.is_dir():
                im = next((cand/s for s in ["images","image","Original"] if (cand/s).exists()), None)
                mk = next((cand/s for s in ["masks","mask","GT","Ground Truth"] if (cand/s).exists()), None)
                if im and mk: out[w] = (im, mk); break
    return out

TEST_SETS = {}   # name -> (img_dir, mask_dir)

# seen sets
if "Kvasir-SEG" in DATA and DATA["Kvasir-SEG"][0]:
    TEST_SETS["Kvasir-SEG (seen)"] = DATA["Kvasir-SEG"]
if DATA.get("CVC-ClinicDB") and DATA["CVC-ClinicDB"][0]:
    TEST_SETS["CVC-ClinicDB (seen)"] = DATA["CVC-ClinicDB"]

# unseen sets — pre-downloaded locally
for nm in ["CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]:
    if DATA.get(nm) and DATA[nm][0]:
        TEST_SETS[nm+" (unseen)"] = DATA[nm]

print("Datasets available for evaluation:")
for k,v in TEST_SETS.items(): print(f"  {k:26s}: {v[0]}")
if len(TEST_SETS) == 1:
    print("\n  NOTE: only Kvasir-SEG found. Training still works; cross-dataset")
    print("  generalization tests are skipped until Kaggle datasets are available.")


# ==== cell ====
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

IMG_EXT = (".png",".jpg",".jpeg",".tif",".tiff",".bmp")

def list_pairs(img_dir, mask_dir):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    masks_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = [(im, masks_by_stem[im.stem]) for im in imgs if im.stem in masks_by_stem]
    if not pairs:  # fall back to positional pairing
        masks = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT])
        pairs = list(zip(imgs, masks))
    return pairs

MEAN = [0.485,0.456,0.406]; STD = [0.229,0.224,0.225]

class PolypDataset(Dataset):
    def __init__(self, pairs, size=224, train=False):
        self.pairs, self.size, self.train = pairs, size, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((self.size,self.size), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((self.size,self.size), Image.NEAREST)
        img = TF.to_tensor(img); msk = TF.to_tensor(msk)
        if self.train:  # light augmentation
            if torch.rand(1) < 0.5: img = TF.hflip(img); msk = TF.hflip(msk)
            if torch.rand(1) < 0.5: img = TF.vflip(img); msk = TF.vflip(msk)
        img = TF.normalize(img, MEAN, STD)
        msk = (msk > 0.5).float()
        return img, msk

# ---- build train/val from seen datasets ----
train_pairs = []
if "Kvasir-SEG" in DATA and DATA["Kvasir-SEG"][0]:
    train_pairs += list_pairs(*DATA["Kvasir-SEG"])
if DATA.get("CVC-ClinicDB") and DATA["CVC-ClinicDB"][0]:
    train_pairs += list_pairs(*DATA["CVC-ClinicDB"])
assert train_pairs, "No training pairs found — check dataset download above."

tr, va = train_test_split(train_pairs, test_size=CFG.VAL_SPLIT, random_state=CFG.SEED)
train_ds = PolypDataset(tr, CFG.IMG_SIZE, train=True)
val_ds   = PolypDataset(va, CFG.IMG_SIZE, train=False)
train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True)
print(f"train pairs: {len(tr)} | val pairs: {len(va)}")


# ==== cell ====
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- differentiable Haar DWT / IDWT ----------
def _haar_filters(device, dtype):
    h = 0.5
    ll = torch.tensor([[h,h],[h,h]], device=device, dtype=dtype)
    lh = torch.tensor([[h,h],[-h,-h]], device=device, dtype=dtype)
    hl = torch.tensor([[h,-h],[h,-h]], device=device, dtype=dtype)
    hh = torch.tensor([[h,-h],[-h,h]], device=device, dtype=dtype)
    return torch.stack([ll,lh,hl,hh]).unsqueeze(1)  # [4,1,2,2]

def dwt2(x):
    B,C,H,W = x.shape
    f = _haar_filters(x.device, x.dtype).repeat(C,1,1,1)  # [4C,1,2,2]
    if H%2: x = F.pad(x,(0,0,0,1)); 
    if W%2: x = F.pad(x,(0,1,0,0))
    out = F.conv2d(x, f, stride=2, groups=C)             # [B,4C,H/2,W/2]
    out = out.view(B, C, 4, out.shape[-2], out.shape[-1])
    return out[:,:,0], out[:,:,1], out[:,:,2], out[:,:,3]  # LL,LH,HL,HH

def idwt2(ll,lh,hl,hh):
    B,C,h,w = ll.shape
    f = _haar_filters(ll.device, ll.dtype).repeat(C,1,1,1)
    coeffs = torch.stack([ll,lh,hl,hh], dim=2).view(B, 4*C, h, w)
    return F.conv_transpose2d(coeffs, f, stride=2, groups=C)

class WaveletFreqModule(nn.Module):
    # High-freq -> boundary enhancement ; Low-freq -> style perturbation (train only).
    def __init__(self, ch, style_prob=0.5):
        super().__init__()
        self.style_prob = style_prob
        self.hf = nn.Sequential(nn.Conv2d(3*ch, 3*ch, 3, padding=1, groups=ch),
                                nn.BatchNorm2d(3*ch), nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(ch, ch, 1)
    def _perturb_ll(self, ll):
        if not self.training: return ll
        B = ll.shape[0]
        do = (torch.rand(B, device=ll.device) < self.style_prob).float().view(B,1,1,1)
        mu = ll.mean(dim=(2,3), keepdim=True); sd = ll.std(dim=(2,3), keepdim=True)+1e-5
        gamma = 1 + 0.3*torch.randn(B,1,1,1, device=ll.device)
        beta  = 0.3*torch.randn(B,1,1,1, device=ll.device)
        ll_p = ((ll-mu)/sd)*(sd*gamma) + (mu+beta*sd)
        return do*ll_p + (1-do)*ll
    def forward(self, x):
        ll,lh,hl,hh = dwt2(x)
        ll = self._perturb_ll(ll)
        B,C = x.shape[:2]
        hf = torch.cat([lh,hl,hh], dim=1)
        hf = self.hf(hf).view(B, C, 3, *lh.shape[-2:])
        lh,hl,hh = hf[:,:,0], hf[:,:,1], hf[:,:,2]
        rec = idwt2(ll,lh,hl,hh)
        rec = F.interpolate(rec, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(rec) + x

class MambaLiteBlock(nn.Module):
    # Lightweight linear-time global-context block (compact SSM-style gated scan).
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.inp  = nn.Conv2d(ch, 2*ch, 1)
        self.dw   = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.a    = nn.Parameter(torch.tensor(0.9))
        self.out  = nn.Conv2d(ch, ch, 1)
    def _scan(self, x):
        B,C,H,W = x.shape
        a = torch.sigmoid(self.a)
        seq = x.flatten(2).transpose(1,2)          # B,(HW),C
        outs = torch.empty_like(seq); h = torch.zeros(B, C, device=x.device)
        for t in range(seq.shape[1]):
            h = a*h + (1-a)*seq[:,t]; outs[:,t] = h
        return outs.transpose(1,2).view(B,C,H,W)
    def forward(self, x):
        r = x; x = self.norm(x)
        u, g = self.inp(x).chunk(2, dim=1)
        u = self.dw(u); u = self._scan(u) * torch.sigmoid(g)
        return r + self.out(u)

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                               nn.Conv2d(co,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self,x): return self.b(x)

class WaveMambaDG(nn.Module):
    def __init__(self, style_prob=0.5, pretrained=True):
        super().__init__()
        try:
            import timm
            self.enc = timm.create_model("mobilenetv3_large_100", features_only=True,
                                         pretrained=pretrained)
            chs = self.enc.feature_info.channels()
            self.timm = True
        except Exception as e:
            print("  timm unavailable, using resnet fallback:", e)
            import torchvision
            m = torchvision.models.resnet34(weights="DEFAULT" if pretrained else None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
            self.l1,self.l2,self.l3,self.l4 = m.layer1,m.layer2,m.layer3,m.layer4
            chs = [64,64,128,256,512]; self.timm = False
        self.wfm = nn.ModuleList([WaveletFreqModule(c, style_prob) for c in chs])
        self.bottleneck = MambaLiteBlock(chs[-1])
        rev = chs[::-1]
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(len(rev)-1):
            self.ups.append(nn.ConvTranspose2d(rev[i], rev[i+1], 2, stride=2))
            self.dec.append(ConvBlock(rev[i+1]*2, rev[i+1]))
        self.mask_head = nn.Conv2d(chs[0], 1, 1)
        self.bnd_head  = nn.Conv2d(chs[0], 1, 1)
    def _features(self, x):
        if self.timm: return self.enc(x)
        x = self.stem(x); f1 = self.l1(x); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        return [x, f1, f2, f3, f4]
    def forward(self, x, task="seg"):
        feats = self._features(x)
        feats = [w(f) for w,f in zip(self.wfm, feats)]
        d = self.bottleneck(feats[-1])
        skips = feats[::-1]
        for i,(up,dec) in enumerate(zip(self.ups, self.dec)):
            d = up(d); s = skips[i+1]
            if d.shape[-2:] != s.shape[-2:]:
                d = F.interpolate(d, size=s.shape[-2:], mode="bilinear", align_corners=False)
            d = dec(torch.cat([d, s], dim=1))
        mask = self.mask_head(d); bnd = self.bnd_head(d)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        bnd  = F.interpolate(bnd,  size=x.shape[-2:], mode="bilinear", align_corners=False)
        # 2-channel seg output keeps it drop-in compatible with common pipelines
        seg2 = torch.cat([-mask, mask], dim=1)
        return {"segmentation": seg2, "mask_logit": mask, "boundary_logit": bnd}

model = WaveMambaDG(style_prob=CFG.STYLE_PROB, pretrained=True).to(CFG.DEVICE)
n_params = sum(p.numel() for p in model.parameters())/1e6
print(f"WaveMamba-DG ready — {n_params:.2f} M parameters")
_ = model(torch.randn(2,3,CFG.IMG_SIZE,CFG.IMG_SIZE, device=CFG.DEVICE))
print("forward OK:", {k:tuple(v.shape) for k,v in _.items()})


# ==== cell ====
import torch, torch.nn as nn, torch.nn.functional as F

def dice_loss(logit, target, eps=1.0):
    p = torch.sigmoid(logit)
    p = p.flatten(1); t = target.flatten(1)
    inter = (p*t).sum(1)
    return (1 - (2*inter+eps)/(p.sum(1)+t.sum(1)+eps)).mean()

def boundary_target(mask):
    # edges via morphological gradient (max-pool trick)
    k = 3
    md_ = F.max_pool2d(mask, k, 1, k//2)
    er = -F.max_pool2d(-mask, k, 1, k//2)
    return (md_ - er).clamp(0,1)

def seg_loss(out, mask):
    ml = out["mask_logit"]
    bce = F.binary_cross_entropy_with_logits(ml, mask)
    dl  = dice_loss(ml, mask)
    bnd = F.binary_cross_entropy_with_logits(out["boundary_logit"], boundary_target(mask))
    return bce + dl + 0.5*bnd

@torch.no_grad()
def seg_metrics(logit, mask, thr=0.5):
    p = (torch.sigmoid(logit) > thr).float()
    t = (mask > 0.5).float()
    tp = (p*t).sum(); fp = (p*(1-t)).sum(); fn = ((1-p)*t).sum(); tn = ((1-p)*(1-t)).sum()
    dice = (2*tp/(2*tp+fp+fn+1e-8)).item()
    iou  = (tp/(tp+fp+fn+1e-8)).item()
    sens = (tp/(tp+fn+1e-8)).item()
    spec = (tn/(tn+fp+1e-8)).item()
    prec = (tp/(tp+fp+1e-8)).item()
    return dict(dice=dice, iou=iou, sensitivity=sens, specificity=spec, precision=prec)


# ==== cell ====
from tqdm.auto import tqdm
import numpy as np, json

_amp = CFG.AMP and CFG.DEVICE.type == "cuda"
opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=_amp)
history = {"train_loss":[], "val_loss":[], "val_dice":[], "val_iou":[]}
best_dice = -1
print(f"[cfg] img={CFG.IMG_SIZE} batch={CFG.BATCH_SIZE} accum={CFG.ACCUM} "
      f"(eff={CFG.BATCH_SIZE*CFG.ACCUM}) amp={_amp} epochs={CFG.EPOCHS} tta={CFG.TTA}", flush=True)

for ep in range(CFG.EPOCHS):
    model.train(); tl = 0; nseen = 0; opt.zero_grad()
    for it,(img, msk) in enumerate(tqdm(train_loader, desc=f"epoch {ep+1}/{CFG.EPOCHS}", leave=False)):
        img, msk = img.to(CFG.DEVICE), msk.to(CFG.DEVICE)
        with torch.cuda.amp.autocast(enabled=_amp):
            out = model(img)
            loss = seg_loss(out, msk) / CFG.ACCUM
        scaler.scale(loss).backward()
        if (it+1) % CFG.ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        tl += loss.item()*CFG.ACCUM*img.size(0); nseen += img.size(0)
        if CFG.MAXITERS and (it+1) >= CFG.MAXITERS: break
    sched.step()

    model.eval(); vl=0; ds=[]; iou=[]
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=_amp):
        for img, msk in val_loader:
            img, msk = img.to(CFG.DEVICE), msk.to(CFG.DEVICE)
            out = model(img); vl += seg_loss(out,msk).item()*img.size(0)
            m = seg_metrics(out["mask_logit"], msk); ds.append(m["dice"]); iou.append(m["iou"])
    tl/=len(train_ds); vl/=len(val_ds); vd=float(np.mean(ds)); vi=float(np.mean(iou))
    history["train_loss"].append(tl); history["val_loss"].append(vl)
    history["val_dice"].append(vd);  history["val_iou"].append(vi)
    print(f"epoch {ep+1:3d} | train {tl:.4f} | val {vl:.4f} | Dice {vd:.4f} | IoU {vi:.4f}")
    if vd > best_dice:
        best_dice = vd; torch.save(model.state_dict(), CFG.OUT_DIR/"wavemamba_dg_best.pt")

json.dump(history, open(CFG.OUT_DIR/"history.json","w"))
print(f"\nBest val Dice: {best_dice:.4f}  (saved to {CFG.OUT_DIR/'wavemamba_dg_best.pt'})")


# ==== cell ====
import numpy as np, pandas as pd

model.load_state_dict(torch.load(CFG.OUT_DIR/"wavemamba_dg_best.pt", map_location=CFG.DEVICE))
model.eval()

_amp_eval = CFG.AMP and CFG.DEVICE.type == "cuda"

def _predict_prob(x):
    # Returns sigmoid probability map. With TTA: mean over identity + h/v flips.
    if not CFG.TTA:
        return torch.sigmoid(model(x)["mask_logit"])
    outs = []
    for f in [lambda t: t, lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-2])]:
        o = model(f(x))["mask_logit"]
        outs.append(torch.sigmoid(f(o)))   # flip is its own inverse
    return torch.stack(outs).mean(0)

@torch.no_grad()
def evaluate_dataset(pairs, name, max_n=None):
    ds = PolypDataset(pairs, CFG.IMG_SIZE, train=False)
    n = len(ds) if max_n is None else min(max_n, len(ds))
    acc = {k:[] for k in ["dice","iou","sensitivity","specificity","precision"]}
    per_dice, per_area, store = [], [], []
    for i in range(n):
        img, msk = ds[i]
        with torch.cuda.amp.autocast(enabled=_amp_eval):
            prob = _predict_prob(img.unsqueeze(0).to(CFG.DEVICE)).float()
        logit = torch.logit(prob.clamp(1e-6, 1-1e-6))   # seg_metrics applies sigmoid
        m = seg_metrics(logit, msk.unsqueeze(0).to(CFG.DEVICE))
        for k in acc: acc[k].append(m[k])
        per_dice.append(m["dice"]); per_area.append(float((msk>0.5).float().mean()))
        if i < 8:  # keep a few for figures
            pr = prob[0,0].cpu().numpy()
            store.append((img, msk.squeeze().cpu().numpy(), pr))
    row = {k: float(np.mean(v)) for k,v in acc.items()}
    row["n"] = n
    return row, np.array(per_dice), np.array(per_area), store

report, PERD, PERA, VIS = {}, {}, {}, {}
for name,(idir,mdir) in TEST_SETS.items():
    pairs = list_pairs(idir, mdir)
    if not pairs: print("  (no pairs) skip", name); continue
    row, pd_, pa_, vis = evaluate_dataset(pairs, name)
    report[name] = row; PERD[name]=pd_; PERA[name]=pa_; VIS[name]=vis
    print(f"{name:26s} | Dice {row['dice']:.4f} | IoU {row['iou']:.4f} | "
          f"Sens {row['sensitivity']:.4f} | Spec {row['specificity']:.4f} | n={row['n']}")

df = pd.DataFrame(report).T[["dice","iou","sensitivity","specificity","precision","n"]]
df = df.round(4)
print("\n================ SEGMENTATION REPORT (per dataset) ================")
print(df.to_string())
def _safe_save_csv(df, path):
    try:
        df.to_csv(path); print("\nsaved ->", path); return
    except PermissionError:
        import time as _t
        alt = str(path).replace(".csv", f"_{_t.strftime('%H%M%S')}.csv")
        df.to_csv(alt); print(f"\n!! {path} was locked (open elsewhere) — saved -> {alt}")
_safe_save_csv(df, CFG.OUT_DIR/"segmentation_report.csv")


# ==== cell ====
import time
model.eval()
params_M = sum(p.numel() for p in model.parameters())/1e6
x = torch.randn(1,3,CFG.IMG_SIZE,CFG.IMG_SIZE, device=CFG.DEVICE)
with torch.no_grad():
    for _ in range(10): model(x)
    if CFG.DEVICE.type=="cuda": torch.cuda.synchronize()
    t=time.time()
    for _ in range(50): model(x)
    if CFG.DEVICE.type=="cuda": torch.cuda.synchronize()
fps = 50/(time.time()-t)
COST = {"params":params_M, "fps":fps}
print(f"Parameters: {params_M:.2f} M | Inference: {fps:.1f} FPS @ {CFG.IMG_SIZE}px on {CFG.DEVICE.type}")


# ==== cell ====
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"figure.facecolor":"white","savefig.facecolor":"white",
                     "savefig.dpi":300,"font.size":11,"axes.titleweight":"bold"})
def _dn(t):
    t=t.detach().cpu().float()
    m=torch.tensor(MEAN).view(3,1,1); s=torch.tensor(STD).view(3,1,1)
    return torch.clamp(t*s+m,0,1).permute(1,2,0).numpy()

# --- 10a training curves ---
h=history; e=np.arange(len(h["train_loss"]))
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].plot(e,h["train_loss"],label="train"); ax[0].plot(e,h["val_loss"],label="val",color="#c0392b")
ax[0].fill_between(e,h["train_loss"],h["val_loss"],color="grey",alpha=.15)
ax[0].set_title("Training convergence"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[1].plot(e,h["val_dice"],color="#59a14f",label="val Dice"); ax[1].plot(e,h["val_iou"],color="#4c78a8",label="val IoU")
ax[1].set_title("Validation Dice / IoU"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.tight_layout(); fig.savefig(CFG.OUT_DIR/"fig_training_curves.png"); plt.show()

# --- 10b qualitative (from first available dataset) ---
name0=list(VIS.keys())[0]; vis=VIS[name0]
n=min(4,len(vis))
fig,ax=plt.subplots(n,4,figsize=(10,2.4*n))
ax=np.atleast_2d(ax)
for r in range(n):
    img,gt,pr=vis[r]
    ov=np.zeros(gt.shape+(3,)); gb=gt>.5; pb=pr>.5
    ov[gb&pb]=[.13,.75,.30]; ov[~gb&pb]=[.9,.2,.2]; ov[gb&~pb]=[.15,.45,.95]
    for c,(d,t,cm) in enumerate([(_dn(img),"Input",None),(gt,"GT","gray"),
                                 (pb,"Prediction","gray"),
                                 (np.clip(_dn(img)*.5+ov*.8,0,1),"Error map",None)]):
        ax[r,c].imshow(d,cmap=cm); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(t)
fig.suptitle(f"Qualitative results — {name0}",y=1.01)
fig.tight_layout(); fig.savefig(CFG.OUT_DIR/"fig_qualitative.png"); plt.show()

# --- 10c per-dataset Dice bar (seen vs unseen) ---
names=list(report.keys()); dices=[report[k]["dice"] for k in names]
colors=["#4c78a8" if "seen" in k and "unseen" not in k else "#c0392b" for k in names]
fig,ax=plt.subplots(figsize=(1.6*len(names)+3,5))
ax.bar(range(len(names)),dices,color=colors,edgecolor="black")
for i,d in enumerate(dices): ax.text(i,d+.005,f"{d:.3f}",ha="center",fontweight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=20,ha="right")
ax.set_ylabel("Dice"); ax.set_ylim(0,1); ax.set_title("Per-dataset Dice (blue=seen, red=unseen)")
fig.tight_layout(); fig.savefig(CFG.OUT_DIR/"fig_per_dataset_dice.png"); plt.show()

# --- 10d wavelet sub-bands (real image) ---
import pywt
img0=_dn(VIS[name0][0][0]); gray=img0.mean(-1)
LL,(LH,HL,HH)=pywt.dwt2(gray,CFG.WAVELET)
fig,ax=plt.subplots(1,5,figsize=(15,3.2))
for a,(t,d,cm) in zip(ax,[("Input",img0,None),("LL",LL,"gray"),("LH",LH,"gray"),("HL",HL,"gray"),("HH",HH,"gray")]):
    a.imshow(d,cmap=cm); a.set_title(t); a.axis("off")
fig.suptitle("Wavelet decomposition of a real frame",y=1.03)
fig.tight_layout(); fig.savefig(CFG.OUT_DIR/"fig_wavelet.png"); plt.show()

# --- 10e per-polyp-size (real) ---
name_seen=[k for k in PERD if "unseen" not in k][0]
a=PERA[name_seen]; d=PERD[name_seen]
bins=[a<.05,(a>=.05)&(a<.15),a>=.15]; labs=["Small","Medium","Large"]
vals=[float(d[b].mean()) if b.sum() else np.nan for b in bins]
fig,ax=plt.subplots(figsize=(7,4.5))
ax.bar(labs,vals,color="#c0392b",edgecolor="black")
for i,v in enumerate(vals):
    if not np.isnan(v): ax.text(i,v+.005,f"{v:.3f}",ha="center",fontweight="bold")
ax.set_ylabel("Dice"); ax.set_ylim(0,1); ax.set_title(f"Dice by polyp size — {name_seen}")
fig.tight_layout(); fig.savefig(CFG.OUT_DIR/"fig_per_size.png"); plt.show()

print("\nAll figures saved to", CFG.OUT_DIR.resolve())

