# Labeled SSFormer-S (baseline) analysis figures: qualitative, error analysis, polyp-size, val-Dice curve.
# Every figure is titled "SSFormer-S (baseline)" so it is unambiguously the baseline, not the proposed model.
import os, sys, json, re
os.chdir(r"D:\AI-Projects\WaveMamba")
import numpy as np, torch
from figlib import *
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from ssformer_pld import SSFormerS

OUT = "outputs/figures_ssformer"; os.makedirs(OUT, exist_ok=True)
def save2(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig); print("SAVED", name)

model = SSFormerS(1).to(DEVICE).eval()
model.load_state_dict(torch.load("outputs/baselines/ssformer_best.pt", map_location=DEVICE, weights_only=True))
@torch.no_grad()
def predict(it):
    x = it.unsqueeze(0).to(DEVICE); outs = []
    for f in [lambda t: t, lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-2])]:
        outs.append(torch.sigmoid(f(model(f(x))[-1])))
    return torch.stack(outs).mean(0)[0, 0].cpu().numpy()
REP = json.load(open("outputs/baselines/ssformer_report.json"))["report"]
TAG = "SSFormer-S (baseline)"

# ---------- Fig 7 analog: qualitative ----------
print("qualitative ...")
TP=(0.16,0.70,0.26); FP=(0.86,0.16,0.16); FN=(0.16,0.36,0.92)
rows=[]
for name,(kind,idir,mdir) in TEST_SETS.items():
    pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(21,len(pairs))).astype(int)
    sc=[]
    for j in idxs:
        it,mt=load_item(*pairs[j]); sc.append((dice_of(predict(it),mt[0].numpy()),it,mt))
    sc.sort(key=lambda t:t[0]); dsc,it,mt=sc[len(sc)//2]
    rows.append((name,kind,denorm(it),mt[0].numpy(),predict(it),dsc))
fig,axes=plt.subplots(len(rows),4,figsize=(7.4,1.85*len(rows)))
cols=["Input","Ground truth","Prediction","Error map"]
for r,(name,kind,img,gt,pr,dsc) in enumerate(rows):
    pb=pr>0.5; gb=gt>0.5; ov=img.copy()
    for m,c in [(gb&pb,TP),((~gb)&pb,FP),(gb&(~pb),FN)]: ov[m]=0.45*ov[m]+0.55*np.array(c)
    for c,d in enumerate([img,gt,pb.astype(float),ov]):
        a=axes[r,c]; a.imshow(d,cmap=None if c in(0,3) else "gray",vmin=0,vmax=1); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#cccccc"); s.set_linewidth(0.6)
        if r==0: a.set_title(cols[c],fontsize=8.5,fontweight="bold",pad=4)
    col=SEEN if kind=="seen" else "#B4611A"
    axes[r,0].set_ylabel(f"{SHORT[name]}\n({kind})",fontsize=8,color=col,fontweight="bold",labelpad=3)
    axes[r,3].text(0.97,0.05,f"Dice {dsc:.2f}",transform=axes[r,3].transAxes,ha="right",va="bottom",
                   fontsize=7.5,color="white",fontweight="bold",bbox=dict(boxstyle="round,pad=0.2",fc=(0,0,0,0.55),ec="none"))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=TP,label="True positive"),Patch(fc=FP,label="False positive"),Patch(fc=FN,label="False negative")],
           loc="lower center",ncol=3,fontsize=8,bbox_to_anchor=(0.5,-0.012))
fig.suptitle(f"Qualitative segmentation — {TAG}",fontsize=9.5,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0.03,1,0.985]); save2(fig,"Figure_qualitative_SSFormerS")

# ---------- Fig 8 analog: error analysis ----------
print("error analysis ...")
names=list(REP.keys()); over=[1-REP[n][5] for n in names]; miss=[1-REP[n][3] for n in names]
edges=np.arange(-25,26,2); centers=(edges[:-1]+edges[1:])/2
acc={"seen":[np.zeros(len(centers)),np.zeros(len(centers))],"unseen":[np.zeros(len(centers)),np.zeros(len(centers))]}
for name,(kind,idir,mdir) in TEST_SETS.items():
    pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(22,len(pairs))).astype(int)
    for j in idxs:
        it,mt=load_item(*pairs[j]); gt=mt[0].numpy()>0.5; pr=predict(it)>0.5
        sd=np.where(gt,-distance_transform_edt(gt),distance_transform_edt(~gt)); err=(pr!=gt).astype(np.float64)
        for b in range(len(centers)):
            m=(sd>=edges[b])&(sd<edges[b+1]); acc[kind][0][b]+=err[m].sum(); acc[kind][1][b]+=m.sum()
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0)); x=np.arange(len(names)); w=0.38
ax[0].bar(x-w/2,over,w,color=RED,edgecolor="#333",lw=0.5,label="Over-seg. (1$-$Prec.)",zorder=3)
ax[0].bar(x+w/2,miss,w,color=BLUE,edgecolor="#333",lw=0.5,label="Missed (1$-$Sens.)",zorder=3)
ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[n] for n in names],rotation=22,ha="right",fontsize=7.5)
ax[0].set_ylabel("Error rate"); ax[0].set_ylim(0,0.40); ax[0].set_title(f"(a) Error composition — {TAG}")
ax[0].legend(loc="upper left"); ax[0].grid(axis="y",color=GRID,lw=0.6); ax[0].set_axisbelow(True); full_box(ax[0])
for kind,col in [("seen",SEEN),("unseen",UNSEEN)]:
    ax[1].plot(centers,acc[kind][0]/np.maximum(acc[kind][1],1),color=col,lw=1.6,ls="-" if kind=="seen" else "--",label=kind)
ax[1].axvline(0,color="#333",lw=0.8,ls=":"); ax[1].text(-1.5,ax[1].get_ylim()[1]*0.99,"GT boundary",fontsize=6.8,color=INK2,ha="right",va="top")
ax[1].set_xlabel("Signed distance from boundary (px)"); ax[1].set_ylabel("Misclassification rate")
ax[1].set_title(f"(b) Errors at boundary — {TAG}"); ax[1].legend(loc="upper right"); ax[1].grid(color=GRID,lw=0.6); ax[1].set_axisbelow(True); full_box(ax[1])
fig.tight_layout(); save2(fig,"Figure_error_SSFormerS")

# ---------- Fig 9 analog: polyp size ----------
print("polyp size ...")
areas=[]; dices=[]; kinds=[]
for name,(kind,idir,mdir) in TEST_SETS.items():
    pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(110,len(pairs))).astype(int)
    for j in idxs:
        it,mt=load_item(*pairs[j]); g=mt[0].numpy()
        if (g>0.5).sum()<1: continue
        areas.append(float((g>0.5).mean())*100); dices.append(dice_of(predict(it),g)); kinds.append(kind)
areas=np.array(areas); dices=np.array(dices); kinds=np.array(kinds)
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0))
for k,col in [("seen",SEEN),("unseen",UNSEEN)]:
    m=kinds==k; ax[0].scatter(areas[m],dices[m],s=9,color=col,alpha=0.30,edgecolors="none",label=f"{k} (n={m.sum()})")
qe=np.quantile(areas,np.linspace(0,1,9)); qc=(qe[:-1]+qe[1:])/2
med=[np.median(dices[(areas>=qe[b])&(areas<=qe[b+1])]) if ((areas>=qe[b])&(areas<=qe[b+1])).sum() else np.nan for b in range(len(qc))]
ax[0].plot(qc,med,color="#222",lw=1.5,marker="o",ms=3,label="median trend")
ax[0].set_xlabel("Polyp area (% of image)"); ax[0].set_ylabel("Per-image Dice"); ax[0].set_ylim(0,1.02); ax[0].set_xlim(0,min(40,areas.max()*1.05))
ax[0].set_title(f"(a) Dice vs. size — {TAG}"); ax[0].legend(loc="lower right",fontsize=7); ax[0].grid(color=GRID,lw=0.6); ax[0].set_axisbelow(True); full_box(ax[0])
bins=[(areas<5),((areas>=5)&(areas<15)),(areas>=15)]; labs=["Small\n(<5%)","Medium\n(5–15%)","Large\n(>15%)"]; xb=np.arange(3); w=0.38
for off,k,col in [(-w/2,"seen",SEEN),(w/2,"unseen",UNSEEN)]:
    vals=[float(dices[b&(kinds==k)].mean()) if (b&(kinds==k)).sum() else np.nan for b in bins]
    ns=[int((b&(kinds==k)).sum()) for b in bins]
    ax[1].bar(xb+off,vals,w,color=col,edgecolor="#333",lw=0.5,zorder=3,label=k)
    for xi,v in zip(xb,vals):
        if not np.isnan(v): ax[1].text(xi+off,v+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=6.8,fontweight="bold")
ax[1].set_xticks(xb); ax[1].set_xticklabels(labs,fontsize=7.5); ax[1].set_ylabel("Mean Dice"); ax[1].set_ylim(0,1.18)
ax[1].set_title(f"(b) Mean Dice by size — {TAG}"); ax[1].legend(loc="upper right",ncol=2); ax[1].grid(axis="y",color=GRID,lw=0.6); ax[1].set_axisbelow(True); full_box(ax[1])
fig.tight_layout(); save2(fig,"Figure_polyp_size_SSFormerS")

# ---------- Fig 3 analog: validation Dice curve (from training log) ----------
print("val-dice curve ...")
vd=[]
for ln in open("_ssformer.log",encoding="utf-8",errors="ignore"):
    mo=re.search(r"epoch\s+(\d+)/\d+ \| val Dice ([0-9.]+)", ln)
    if mo: vd.append((int(mo.group(1)),float(mo.group(2))))
if vd:
    vd=sorted(dict(vd).items()); e=[a for a,_ in vd]; d=[b for _,b in vd]
    fig,ax=plt.subplots(figsize=(4.6,3.0))
    ax.plot(e,d,color=BLUE,lw=1.8,marker="o",ms=2.5)
    bi=int(np.argmax(d)); ax.scatter([e[bi]],[d[bi]],color=BLUE,s=24,zorder=6,edgecolor="white",lw=0.6)
    ax.annotate(f"best {d[bi]:.3f} (ep {e[bi]})",(e[bi],d[bi]),textcoords="offset points",xytext=(-6,8),ha="right",fontsize=7.5,color=INK)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Dice"); ax.set_title(f"Validation Dice — {TAG}")
    ax.grid(axis="y",color=GRID,lw=0.6); ax.set_axisbelow(True); full_box(ax)
    fig.tight_layout(); save2(fig,"Figure_valdice_SSFormerS")
print("SSFORMER_FIGS_DONE")
