# Retrain SSFormer-S (baseline) with FULL per-epoch history (train/val loss, val Dice/IoU) -> 2-panel figure.
# Writes to *separate* files (ssformer_hist_*) so the canonical reported SSFormer results are untouched.
import os, json, time
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from figlib import *
import matplotlib.pyplot as plt
from ssformer_pld import SSFormerS

EPOCHS=60; BATCH,ACCUM,LR,WD=4,2,1e-4,1e-4; TAG="SSFormer-S (baseline)"
seen=list_pairs(*TEST_SETS["Kvasir-SEG"][1:])+list_pairs(*TEST_SETS["CVC-ClinicDB"][1:])
tr,va=train_test_split(seen,test_size=0.2,random_state=42)
print(f"train={len(tr)} val={len(va)}",flush=True)
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
def structure_loss(logit,mask):
    weit=1+5*torch.abs(F.avg_pool2d(mask,31,1,15)-mask)
    wbce=F.binary_cross_entropy_with_logits(logit,mask,reduction='none')
    wbce=(weit*wbce).sum((2,3))/weit.sum((2,3))
    pred=torch.sigmoid(logit); inter=((pred*mask)*weit).sum((2,3)); union=((pred+mask)*weit).sum((2,3))
    return (wbce+(1-(inter+1)/(union-inter+1))).mean()

def main():
    model=SSFormerS(1).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=torch.cuda.amp.GradScaler()
    tl=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(DS(va,False),batch_size=BATCH,shuffle=False,num_workers=0,pin_memory=True)
    hist={"train_loss":[],"val_loss":[],"val_dice":[],"val_iou":[]}; best=-1
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad(); t0=time.time(); tl_sum=0; n=0
        for it,(img,m) in enumerate(tl):
            img,m=img.to(DEVICE),m.to(DEVICE)
            with torch.cuda.amp.autocast():
                loss=structure_loss(model(img)[-1],m)/ACCUM
            scaler.scale(loss).backward()
            if (it+1)%ACCUM==0: scaler.step(opt); scaler.update(); opt.zero_grad()
            tl_sum+=loss.item()*ACCUM*img.size(0); n+=img.size(0)
        sch.step()
        model.eval(); vl_sum=0; vn=0; ds=[]; iou=[]
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img,m in vl:
                img,m=img.to(DEVICE),m.to(DEVICE); logit=model(img)[-1]
                vl_sum+=structure_loss(logit,m).item()*img.size(0); vn+=img.size(0)
                p=(torch.sigmoid(logit)>0.5).float()
                for k in range(p.shape[0]):
                    pp=p[k,0]; tt=m[k,0]; inter=(pp*tt).sum().item()
                    ds.append((2*inter+1)/(pp.sum().item()+tt.sum().item()+1))
                    uni=pp.sum().item()+tt.sum().item()-inter; iou.append((inter+1)/(uni+1))
        hist["train_loss"].append(tl_sum/n); hist["val_loss"].append(vl_sum/vn)
        hist["val_dice"].append(float(np.mean(ds))); hist["val_iou"].append(float(np.mean(iou)))
        print(f"[ssformer_hist] epoch {ep+1:3d}/{EPOCHS} | train {hist['train_loss'][-1]:.4f} | val {hist['val_loss'][-1]:.4f} | Dice {hist['val_dice'][-1]:.4f} | IoU {hist['val_iou'][-1]:.4f} | {time.time()-t0:.0f}s",flush=True)
        if hist["val_dice"][-1]>best: best=hist["val_dice"][-1]; torch.save(model.state_dict(),"outputs/baselines/ssformer_hist_best.pt")
    json.dump(hist,open("outputs/baselines/ssformer_history.json","w"))

    # 2-panel figure (matches WaveLite-DG Fig 3 style)
    e=np.arange(1,len(hist["train_loss"])+1)
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
    ax[0].plot(e,hist["train_loss"],color=BLUE,lw=1.6,ls="-",label="Train")
    ax[0].plot(e,hist["val_loss"],color=UNSEEN,lw=1.6,ls="--",label="Validation")
    ax[0].fill_between(e,hist["train_loss"],hist["val_loss"],color=MUTED,alpha=0.12,lw=0)
    ax[0].set_title(f"(a) Training convergence — {TAG}"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].legend(loc="upper right"); ax[0].set_xlim(1,len(e)); ax[0].set_ylim(0,None)
    ax[1].plot(e,hist["val_dice"],color=BLUE,lw=1.6,ls="-",label="Dice")
    ax[1].plot(e,hist["val_iou"],color=LBLUE,lw=1.6,ls="--",label="IoU")
    bi=int(np.argmax(hist["val_dice"]))
    ax[1].scatter([e[bi]],[hist["val_dice"][bi]],color=BLUE,s=22,zorder=6,edgecolor="white",lw=0.6)
    ax[1].annotate(f"best Dice = {hist['val_dice'][bi]:.3f} (epoch {e[bi]})",(e[bi],hist["val_dice"][bi]),
                   textcoords="offset points",xytext=(-8,8),ha="right",fontsize=7.5,color=INK,
                   arrowprops=dict(arrowstyle="-",color=INK2,lw=0.6))
    ax[1].set_title(f"(b) Validation Dice / IoU — {TAG}"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score")
    ax[1].set_ylim(0.55,1.0); ax[1].set_xlim(1,len(e)); ax[1].legend(loc="lower right")
    for a in ax: a.grid(axis="y",color=GRID,lw=0.6); a.set_axisbelow(True); full_box(a)
    fig.tight_layout()
    os.makedirs("outputs/figures_ssformer",exist_ok=True)
    fig.savefig("outputs/figures_ssformer/Figure_training_SSFormerS.pdf",bbox_inches="tight")
    fig.savefig("outputs/figures_ssformer/Figure_training_SSFormerS.png",dpi=320,bbox_inches="tight")
    plt.close(fig); print("SAVED Figure_training_SSFormerS"); print("SSFORMER_HIST_DONE",flush=True)

if __name__=="__main__": main()
