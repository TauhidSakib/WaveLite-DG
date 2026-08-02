# Fair baseline harness. Usage: python train_baseline.py <arch> [epochs]
# Same seed-42 split, 352px, AdamW 1e-4 cosine, BCE+Dice, AMP, best-ckpt on val Dice, TTA eval.
# archs: unet | ours (eval-only) | pranet | polyppvt | ssformer (added incrementally)
import sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from figlib import (DEVICE, IMG, MEAN, STD, list_pairs, load_item, TEST_SETS, WaveMambaDG)

ARCH   = sys.argv[1] if len(sys.argv) > 1 else "unet"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
_BCFG={"unet":(4,2),"pranet":(4,2),"polyppvt":(4,2),"ssformer":(4,2)}
BATCH, ACCUM = _BCFG.get(ARCH,(4,2)); LR, WD = 1e-4, 1e-4
OUT = f"outputs/baselines"; import os; os.makedirs(OUT, exist_ok=True)
_amp = DEVICE.type == "cuda"

# ---------- fair split (identical to WaveMamba-DG training) ----------
seen = list_pairs(*TEST_SETS["Kvasir-SEG"][1:]) + list_pairs(*TEST_SETS["CVC-ClinicDB"][1:])
tr, va = train_test_split(seen, test_size=0.2, random_state=42)
va_kv = [p for p in va if "Kvasir-SEG" in str(p[0])]
va_cl = [p for p in va if "cvcclinicdb" in str(p[0])]
TESTSETS = {  # name -> (kind, pairs)
    "Kvasir-SEG":   ("seen",   va_kv),
    "CVC-ClinicDB": ("seen",   va_cl),
    "CVC-ColonDB":  ("unseen", list_pairs(*TEST_SETS["CVC-ColonDB"][1:])),
    "ETIS-LaribPolypDB": ("unseen", list_pairs(*TEST_SETS["ETIS-LaribPolypDB"][1:])),
    "CVC-300":      ("unseen", list_pairs(*TEST_SETS["CVC-300"][1:])),
}
print(f"[{ARCH}] train={len(tr)}  val(seen-test): Kvasir={len(va_kv)} ClinicDB={len(va_cl)}", flush=True)

class DS(Dataset):
    def __init__(self, pairs, train=False): self.p=pairs; self.train=train
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        ip, mp = self.p[i]
        img = Image.open(ip).convert("RGB").resize((IMG,IMG), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((IMG,IMG), Image.NEAREST)
        img = TF.to_tensor(img); msk = (TF.to_tensor(msk) > 0.5).float()
        if self.train:
            if torch.rand(1) < 0.5: img = TF.hflip(img); msk = TF.hflip(msk)
            if torch.rand(1) < 0.5: img = TF.vflip(img); msk = TF.vflip(msk)
        return TF.normalize(img, MEAN, STD), msk

def build(arch):
    if arch == "unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "pranet":
        import sys as _s; _s.path.insert(0, "_baselines/PraNet")
        from lib.PraNet_Res2Net import PraNet
        return PraNet()
    if arch == "polyppvt":
        import sys as _s; _s.path.insert(0, "_baselines/Polyp-PVT")
        from lib.pvt import PolypPVT
        return PolypPVT()
    if arch == "ssformer":
        from ssformer_pld import SSFormerS
        return SSFormerS(class_num=1)
    raise ValueError(f"arch {arch} not yet wired")

def forward_logit(model, x):
    o = model(x)
    if isinstance(o, dict): o = o["mask_logit"]
    if isinstance(o, (list, tuple)): o = o[-1]   # PraNet: finest map (lateral_map_2)
    return o

def dice_loss(logit, t, eps=1.0):
    p = torch.sigmoid(logit).flatten(1); t = t.flatten(1)
    return (1 - (2*(p*t).sum(1)+eps)/(p.sum(1)+t.sum(1)+eps)).mean()
def seg_loss(logit, t):
    return F.binary_cross_entropy_with_logits(logit, t) + dice_loss(logit, t)
def structure_loss(logit, mask):   # PraNet's boundary-weighted BCE + weighted IoU
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask,31,1,15)-mask)
    wbce = F.binary_cross_entropy_with_logits(logit, mask, reduction='none')
    wbce = (weit*wbce).sum((2,3))/weit.sum((2,3))
    pred = torch.sigmoid(logit)
    inter=((pred*mask)*weit).sum((2,3)); union=((pred+mask)*weit).sum((2,3))
    wiou=1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def metrics(pr, gt, thr=0.5):
    p=(pr>thr).astype(np.float64); t=(gt>0.5).astype(np.float64)
    tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum(); tn=((1-p)*(1-t)).sum()
    return [2*tp/(2*tp+fp+fn+1e-8), tp/(tp+fp+fn+1e-8), tp/(tp+fn+1e-8), tn/(tn+fp+1e-8), tp/(tp+fp+1e-8)]

@torch.no_grad()
def tta_prob(model, it):
    x=it.unsqueeze(0).to(DEVICE); outs=[]
    for f in [lambda t:t, lambda t:torch.flip(t,[-1]), lambda t:torch.flip(t,[-2])]:
        outs.append(torch.sigmoid(f(forward_logit(model, f(x)))))
    return torch.stack(outs).mean(0)[0,0].cpu().numpy()

@torch.no_grad()
def evaluate(model):
    model.eval(); rep={}; perimg={}
    for name,(kind,pairs) in TESTSETS.items():
        acc=np.zeros(5); ds=[]
        for ip,mp in pairs:
            it,mt=load_item(ip,mp); pr=tta_prob(model,it); gt=mt[0].numpy()
            m=metrics(pr,gt); acc+=np.array(m); ds.append(m[0])
        acc/=len(pairs); rep[name]=[kind]+[round(float(v),4) for v in acc]+[len(pairs)]; perimg[name]=ds
    return rep, perimg

def main():
    if ARCH == "ours":
        model=WaveMambaDG(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load("outputs/wavemamba_dg_best.pt",map_location=DEVICE,weights_only=True))
        rep,perimg=evaluate(model)
    else:
        model=build(ARCH).to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        scaler=torch.cuda.amp.GradScaler(enabled=_amp)
        tl=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
        vl=DataLoader(DS(va,False),batch_size=BATCH,shuffle=False,num_workers=0,pin_memory=True)
        best=-1
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad(); t0=time.time()
            for it,(img,msk) in enumerate(tl):
                img,msk=img.to(DEVICE),msk.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=_amp):
                    raw=model(img)
                    if isinstance(raw,(list,tuple)):
                        loss=sum(structure_loss(m,msk) for m in raw)/ACCUM   # deep supervision
                    else:
                        loss=seg_loss(raw,msk)/ACCUM
                scaler.scale(loss).backward()
                if (it+1)%ACCUM==0: scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            model.eval(); ds=[]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=_amp):
                for img,msk in vl:
                    img=img.to(DEVICE)
                    p=(torch.sigmoid(forward_logit(model,img))>0.5).float().cpu()
                    for k in range(p.shape[0]):
                        pp=p[k,0].numpy(); tt=msk[k,0].numpy()
                        inter=(pp*tt).sum(); ds.append((2*inter+1)/(pp.sum()+tt.sum()+1))
            vd=float(np.mean(ds))
            print(f"[{ARCH}] epoch {ep+1:3d}/{EPOCHS} | val Dice {vd:.4f} | {time.time()-t0:.0f}s",flush=True)
            if vd>best: best=vd; torch.save(model.state_dict(),f"{OUT}/{ARCH}_best.pt")
        model.load_state_dict(torch.load(f"{OUT}/{ARCH}_best.pt",map_location=DEVICE,weights_only=True))
        rep,perimg=evaluate(model)
    json.dump({"arch":ARCH,"report":rep}, open(f"{OUT}/{ARCH}_report.json","w"), indent=1)
    json.dump(perimg, open(f"{OUT}/{ARCH}_perimage.json","w"))
    print(f"\n==== {ARCH} RESULTS (fair split, TTA) ====",flush=True)
    for n,v in rep.items(): print(f"  {n:20s} {v[0]:6s} Dice {v[1]:.4f} IoU {v[2]:.4f} n={v[6]}",flush=True)
    print(f"{ARCH.upper()}_DONE",flush=True)

if __name__=="__main__": main()
