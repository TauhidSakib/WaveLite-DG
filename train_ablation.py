# Ablation study for WaveLite-DG. Trains 4 cumulative variants on the fair split (100 ep/352, TTA eval);
# the 5th row (full model) reuses the existing checkpoint's numbers. Saves outputs/ablation/<cfg>.json.
import os, json, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from figlib import (DEVICE, IMG, MEAN, STD, list_pairs, load_item, TEST_SETS, WaveMambaDG)

EPOCHS = int(os.environ.get("ABL_EPOCHS", 100)); BATCH, ACCUM = 4, 2
OUT = "outputs/ablation"; os.makedirs(OUT, exist_ok=True)
UNSEEN = ["CVC-ColonDB","ETIS-LaribPolypDB","CVC-300"]
# cfg -> (use_wfm, style_prob, use_ssm, boundary_weight)
CFGS = {
    "base": (False, 0.0, False, 0.0),   # encoder + decoder only
    "bnd":  (False, 0.0, False, 0.5),   # + boundary head
    "hf":   (True,  0.0, False, 0.5),   # + WFM high-freq path (no LL perturbation)
    "llp":  (True,  0.5, False, 0.5),   # + LL style perturbation (WFM full, no SSM)
    # "full" = existing model (use_wfm=T, style_prob=.5, use_ssm=T, bnd=.5) -> reuse reported numbers
}

seen = list_pairs(*TEST_SETS["Kvasir-SEG"][1:]) + list_pairs(*TEST_SETS["CVC-ClinicDB"][1:])
tr, va = train_test_split(seen, test_size=0.2, random_state=42)
TESTS = {n: list_pairs(*TEST_SETS[n][1:]) for n in UNSEEN}
print(f"train={len(tr)} val={len(va)}", flush=True)

class DS(Dataset):
    def __init__(s,p,train=False): s.p=p; s.train=train
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        ip,mp=s.p[i]
        img=Image.open(ip).convert("RGB").resize((IMG,IMG),Image.BILINEAR)
        msk=Image.open(mp).convert("L").resize((IMG,IMG),Image.NEAREST)
        img=TF.to_tensor(img); msk=(TF.to_tensor(msk)>0.5).float()
        if s.train:
            if torch.rand(1)<0.5: img=TF.hflip(img); msk=TF.hflip(msk)
            if torch.rand(1)<0.5: img=TF.vflip(img); msk=TF.vflip(msk)
        return TF.normalize(img,MEAN,STD), msk

def dice_loss(l,t,eps=1.0):
    p=torch.sigmoid(l).flatten(1); t=t.flatten(1)
    return (1-(2*(p*t).sum(1)+eps)/(p.sum(1)+t.sum(1)+eps)).mean()
def bnd_target(m):
    md=F.max_pool2d(m,3,1,1); er=-F.max_pool2d(-m,3,1,1); return (md-er).clamp(0,1)
def loss_fn(out,m,bw):
    l=F.binary_cross_entropy_with_logits(out["mask_logit"],m)+dice_loss(out["mask_logit"],m)
    if bw>0: l=l+bw*F.binary_cross_entropy_with_logits(out["boundary_logit"],bnd_target(m))
    return l
def dice_of(pr,gt,thr=0.5):
    p=(pr>thr).astype(np.float32); t=(gt>0.5).astype(np.float32)
    return float((2*(p*t).sum()+1)/(p.sum()+t.sum()+1))
@torch.no_grad()
def tta(model,it):
    x=it.unsqueeze(0).to(DEVICE); outs=[]
    for f in [lambda t:t,lambda t:torch.flip(t,[-1]),lambda t:torch.flip(t,[-2])]:
        outs.append(torch.sigmoid(f(model(f(x))["mask_logit"])))
    return torch.stack(outs).mean(0)[0,0].cpu().numpy()

def run_cfg(name):
    uw,sp,us,bw = CFGS[name]
    print(f"\n===== ABLATION {name}: use_wfm={uw} style_prob={sp} use_ssm={us} bnd_w={bw} =====", flush=True)
    model=WaveMambaDG(style_prob=sp,pretrained=True,use_wfm=uw,use_ssm=us).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=torch.cuda.amp.GradScaler()
    tl=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(DS(va,False),batch_size=BATCH,shuffle=False,num_workers=0,pin_memory=True)
    best=-1; bestpath=f"{OUT}/{name}_best.pt"
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad(); t0=time.time()
        for it,(img,m) in enumerate(tl):
            img,m=img.to(DEVICE),m.to(DEVICE)
            with torch.cuda.amp.autocast():
                l=loss_fn(model(img),m,bw)/ACCUM
            scaler.scale(l).backward()
            if (it+1)%ACCUM==0: scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        model.eval(); ds=[]
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img,m in vl:
                p=(torch.sigmoid(model(img.to(DEVICE))["mask_logit"])>0.5).float().cpu()
                for k in range(p.shape[0]):
                    pp=p[k,0].numpy(); tt=m[k,0].numpy(); ds.append((2*(pp*tt).sum()+1)/(pp.sum()+tt.sum()+1))
        vd=float(np.mean(ds))
        if ep%10==0 or ep==EPOCHS-1:
            print(f"[{name}] epoch {ep+1:3d}/{EPOCHS} val {vd:.4f} {time.time()-t0:.0f}s", flush=True)
        if vd>best: best=vd; torch.save(model.state_dict(),bestpath)
    model.load_state_dict(torch.load(bestpath,map_location=DEVICE,weights_only=True)); model.eval()
    rep={}
    for n,pairs in TESTS.items():
        d=[dice_of(tta(model,it),mt[0].numpy()) for it,mt in (load_item(*p) for p in pairs)]
        rep[n]=round(float(np.mean(d)),4)
    rep["mean_unseen"]=round(float(np.mean([rep[n] for n in UNSEEN])),4)
    rep["params_M"]=round(sum(p.numel() for p in model.parameters())/1e6,2)
    json.dump({"cfg":name,"flags":CFGS[name],"report":rep}, open(f"{OUT}/{name}.json","w"), indent=1)
    print(f"[{name}] RESULT {rep}", flush=True)

if __name__=="__main__":
    todo = sys.argv[1:] if len(sys.argv)>1 else ["base","bnd","hf","llp"]
    for c in todo: run_cfg(c)
    print("ABLATION_ALL_DONE", flush=True)
