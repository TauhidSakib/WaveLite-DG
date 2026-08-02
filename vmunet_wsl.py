# VM-UNet fair baseline, run inside WSL (mamba-ssm GPU). Same seed-42 split / 352px / TTA as the others.
import sys, json, time, os
sys.path.insert(0, "/mnt/d/AI-Projects/WaveMamba/_baselines/VM-UNet")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from models.vmunet.vmunet import VMUNet

ROOT="/mnt/d/AI-Projects/WaveMamba"
DEVICE=torch.device("cuda"); IMG=352; MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 60
BATCH,ACCUM,LR,WD=2,4,1e-4,1e-4
EXT=(".png",".jpg",".jpeg",".tif",".tiff",".bmp")
def lp(img_dir,mask_dir):
    from pathlib import Path
    img_dir,mask_dir=Path(img_dir),Path(mask_dir)
    imgs=sorted([p for p in img_dir.iterdir() if p.suffix.lower() in EXT])
    mby={p.stem:p for p in mask_dir.iterdir() if p.suffix.lower() in EXT}
    pairs=[(im,mby[im.stem]) for im in imgs if im.stem in mby]
    if not pairs:
        masks=sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in EXT]); pairs=list(zip(imgs,masks))
    return pairs
D=f"{ROOT}/data"
TEST_SETS={
 "Kvasir-SEG":("seen",f"{D}/Kvasir-SEG/images",f"{D}/Kvasir-SEG/masks"),
 "CVC-ClinicDB":("seen",f"{D}/_kaggle/cvcclinicdb/PNG/Original",f"{D}/_kaggle/cvcclinicdb/PNG/Ground Truth"),
 "CVC-ColonDB":("unseen",f"{D}/_kaggle/cvc-colondb/CVC-ColonDB/images",f"{D}/_kaggle/cvc-colondb/CVC-ColonDB/masks"),
 "ETIS-LaribPolypDB":("unseen",f"{D}/_kaggle/etis-larib/images",f"{D}/_kaggle/etis-larib/masks"),
 "CVC-300":("unseen",f"{D}/_kaggle/cvc-300/CVC-300/images",f"{D}/_kaggle/cvc-300/CVC-300/masks"),
}
seen=lp(*TEST_SETS["Kvasir-SEG"][1:])+lp(*TEST_SETS["CVC-ClinicDB"][1:])
tr,va=train_test_split(seen,test_size=0.2,random_state=42)
va_kv=[p for p in va if "Kvasir-SEG" in str(p[0])]; va_cl=[p for p in va if "cvcclinicdb" in str(p[0])]
TESTS={"Kvasir-SEG":("seen",va_kv),"CVC-ClinicDB":("seen",va_cl),
       "CVC-ColonDB":("unseen",lp(*TEST_SETS["CVC-ColonDB"][1:])),
       "ETIS-LaribPolypDB":("unseen",lp(*TEST_SETS["ETIS-LaribPolypDB"][1:])),
       "CVC-300":("unseen",lp(*TEST_SETS["CVC-300"][1:]))}
print(f"[vmunet] train={len(tr)} valKv={len(va_kv)} valCl={len(va_cl)}",flush=True)

def load_item(ip,mp):
    img=Image.open(ip).convert("RGB").resize((IMG,IMG),Image.BILINEAR)
    msk=Image.open(mp).convert("L").resize((IMG,IMG),Image.NEAREST)
    return TF.normalize(TF.to_tensor(img),MEAN,STD),(TF.to_tensor(msk)>0.5).float()
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
        return TF.normalize(img,MEAN,STD),msk

def dice_loss(l,t,eps=1.0):
    p=torch.sigmoid(l).flatten(1); t=t.flatten(1)
    return (1-(2*(p*t).sum(1)+eps)/(p.sum(1)+t.sum(1)+eps)).mean()
def seg_loss(l,t): return F.binary_cross_entropy_with_logits(l,t)+dice_loss(l,t)
def logit_of(model,x): return model.vmunet(x)   # raw logits (bypass wrapper sigmoid)
def metrics(pr,gt,thr=0.5):
    p=(pr>thr).astype(np.float64); t=(gt>0.5).astype(np.float64)
    tp=(p*t).sum();fp=(p*(1-t)).sum();fn=((1-p)*t).sum();tn=((1-p)*(1-t)).sum()
    return [2*tp/(2*tp+fp+fn+1e-8),tp/(tp+fp+fn+1e-8),tp/(tp+fn+1e-8),tn/(tn+fp+1e-8),tp/(tp+fp+1e-8)]
@torch.no_grad()
def tta_prob(model,it):
    x=it.unsqueeze(0).to(DEVICE); outs=[]
    for f in [lambda t:t,lambda t:torch.flip(t,[-1]),lambda t:torch.flip(t,[-2])]:
        outs.append(torch.sigmoid(f(logit_of(model,f(x)))))
    return torch.stack(outs).mean(0)[0,0].cpu().numpy()
@torch.no_grad()
def evaluate(model):
    model.eval(); rep={}; per={}
    for name,(kind,pairs) in TESTS.items():
        acc=np.zeros(5); ds=[]
        for ip,mp in pairs:
            it,mt=load_item(ip,mp); pr=tta_prob(model,it); gt=mt[0].numpy()
            m=metrics(pr,gt); acc+=np.array(m); ds.append(m[0])
        acc/=len(pairs); rep[name]=[kind]+[round(float(v),4) for v in acc]+[len(pairs)]; per[name]=ds
    return rep,per

def main():
    model=VMUNet(num_classes=1,input_channels=3,depths=[2,2,9,2],depths_decoder=[2,9,2,2],
                 drop_path_rate=0.2,load_ckpt_path=f"{ROOT}/_baselines/VM-UNet/pretrained_weights/vmamba_tiny_e292.pth").to(DEVICE)
    model.load_from()
    print("[vmunet] params(M)",round(sum(p.numel() for p in model.parameters())/1e6,2),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=torch.cuda.amp.GradScaler()
    tl=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(DS(va,False),batch_size=BATCH,shuffle=False,num_workers=0,pin_memory=True)
    best=-1; OUT=f"{ROOT}/outputs/baselines"; os.makedirs(OUT,exist_ok=True)
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad(); t0=time.time()
        for it,(img,msk) in enumerate(tl):
            img,msk=img.to(DEVICE),msk.to(DEVICE)
            with torch.cuda.amp.autocast():
                loss=seg_loss(logit_of(model,img),msk)/ACCUM
            scaler.scale(loss).backward()
            if (it+1)%ACCUM==0: scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()
        model.eval(); ds=[]
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img,msk in vl:
                p=(torch.sigmoid(logit_of(model,img.to(DEVICE)))>0.5).float().cpu()
                for k in range(p.shape[0]):
                    pp=p[k,0].numpy(); tt=msk[k,0].numpy(); ds.append((2*(pp*tt).sum()+1)/(pp.sum()+tt.sum()+1))
        vd=float(np.mean(ds)); print(f"[vmunet] epoch {ep+1:3d}/{EPOCHS} | val Dice {vd:.4f} | {time.time()-t0:.0f}s",flush=True)
        if vd>best: best=vd; torch.save(model.state_dict(),f"{OUT}/vmunet_best.pt")
    model.load_state_dict(torch.load(f"{OUT}/vmunet_best.pt",map_location=DEVICE,weights_only=True))
    rep,per=evaluate(model)
    json.dump({"arch":"vmunet","report":rep},open(f"{OUT}/vmunet_report.json","w"),indent=1)
    json.dump(per,open(f"{OUT}/vmunet_perimage.json","w"))
    print("\n==== vmunet RESULTS (fair split, TTA) ====",flush=True)
    for n,v in rep.items(): print(f"  {n:20s} {v[0]:6s} Dice {v[1]:.4f} IoU {v[2]:.4f} n={v[6]}",flush=True)
    print("VMUNET_DONE",flush=True)

if __name__=="__main__": main()
