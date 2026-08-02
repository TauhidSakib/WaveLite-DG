# One-at-a-time genuine figure generator.  Usage: python mkfig.py <figure_name>
import sys, json
import numpy as np
from figlib import *

def fig_training():
    h = json.load(open("outputs/history.json"))
    e = np.arange(1, len(h["train_loss"])+1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # (a) loss
    ax[0].plot(e, h["train_loss"], color=BLUE,   lw=1.6, ls="-",  label="Train")
    ax[0].plot(e, h["val_loss"],   color=UNSEEN, lw=1.6, ls="--", label="Validation")
    ax[0].fill_between(e, h["train_loss"], h["val_loss"], color=MUTED, alpha=0.12, lw=0)
    ax[0].set_title("(a) Training convergence"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].legend(loc="upper right"); ax[0].set_xlim(1, len(e)); ax[0].set_ylim(0, None)
    # (b) dice / iou
    ax[1].plot(e, h["val_dice"], color=BLUE,  lw=1.6, ls="-",  label="Dice")
    ax[1].plot(e, h["val_iou"],  color=LBLUE, lw=1.6, ls="--", label="IoU")
    bi = int(np.argmax(h["val_dice"]))
    ax[1].scatter([e[bi]], [h["val_dice"][bi]], color=BLUE, s=22, zorder=6, edgecolor="white", lw=0.6)
    ax[1].annotate(f"best Dice = {h['val_dice'][bi]:.3f} (epoch {e[bi]})",
                   (e[bi], h["val_dice"][bi]), textcoords="offset points", xytext=(-8, 8),
                   ha="right", fontsize=7.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax[1].set_title("(b) Validation Dice / IoU"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score")
    ax[1].set_ylim(0.55, 1.0); ax[1].set_xlim(1, len(e)); ax[1].legend(loc="lower right")
    for a in ax:
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True); full_box(a)
    fig.tight_layout()
    save(fig, "fig_training_convergence")

def fig_perdataset():
    names = list(REPORT.keys())
    dice  = [REPORT[n][1] for n in names]
    iou   = [REPORT[n][2] for n in names]
    kinds = [REPORT[n][0] for n in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    # unseen background band
    ax.axvspan(1.5, len(names)-0.4, color=UNSEEN, alpha=0.05, lw=0)
    ax.axvline(1.5, color=BASE, lw=0.8, ls=(0, (4, 3)))
    for xi, d, i, k in zip(x, dice, iou, kinds):
        cD = SEEN if k == "seen" else UNSEEN
        cI = SEEN_L if k == "seen" else UNSEEN_L
        ax.bar(xi-w/2, d, w, color=cD, edgecolor="#333333", lw=0.5, zorder=3)
        ax.bar(xi+w/2, i, w, color=cI, edgecolor="#333333", lw=0.5, zorder=3)
        ax.text(xi-w/2, d+0.012, f"{d:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(xi+w/2, i+0.012, f"{i:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(0.5, 1.10, "Seen (training centres)", ha="center", fontsize=8, fontweight="bold", color=SEEN)
    ax.text(3.0, 1.10, "Unseen (cross-centre)", ha="center", fontsize=8, fontweight="bold", color=UNSEEN_D if (UNSEEN_D:="#B4611A") else UNSEEN)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.20); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_xlim(-0.6, len(names)-0.4)
    # metric legend via proxies (neutral, grayscale-safe: dark=Dice, light=IoU)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#555555", ec="#333333", lw=0.5, label="Dice"),
                       Patch(fc="#bfbfbf", ec="#333333", lw=0.5, label="IoU")],
              loc="lower left", ncol=2, handlelength=1.1, columnspacing=1.0)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); full_box(ax)
    fig.tight_layout(); save(fig, "fig2_per_dataset_metrics")

def fig_qualitative():
    # representative (median-Dice) real case per dataset; real predictions
    TP=(0.16,0.70,0.26); FP=(0.86,0.16,0.16); FN=(0.16,0.36,0.92)
    rows=[]
    for name,(kind,idir,mdir) in TEST_SETS.items():
        pairs=list_pairs(idir,mdir)
        idxs=np.linspace(0,len(pairs)-1,min(21,len(pairs))).astype(int)
        scored=[]
        for j in idxs:
            it,mt=load_item(*pairs[j]); scored.append((dice_of(predict(it),mt[0].numpy()),j,it,mt))
        scored.sort(key=lambda t:t[0]); dsc,j,it,mt=scored[len(scored)//2]
        rows.append((name,kind,denorm(it),mt[0].numpy(),predict(it),dsc))
    coltitles=["Input","Ground truth","Prediction","Error map"]
    fig,axes=plt.subplots(len(rows),4,figsize=(7.4,1.85*len(rows)))
    for r,(name,kind,img,gt,pr,dsc) in enumerate(rows):
        pb=pr>0.5; gb=gt>0.5
        ov=img.copy()
        for m,c in [(gb&pb,TP),((~gb)&pb,FP),(gb&(~pb),FN)]:
            ov[m]=0.45*ov[m]+0.55*np.array(c)
        for c,d in enumerate([img,gt,pb.astype(float),ov]):
            a=axes[r,c]; a.imshow(d,cmap=None if c in (0,3) else "gray",vmin=0,vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_color("#cccccc"); s.set_linewidth(0.6)
            if r==0: a.set_title(coltitles[c],fontsize=8.5,fontweight="bold",pad=4)
        col=SEEN if kind=="seen" else "#B4611A"
        axes[r,0].set_ylabel(f"{SHORT[name]}\n({kind})",fontsize=8,color=col,fontweight="bold",rotation=90,labelpad=3)
        axes[r,3].text(0.97,0.05,f"Dice {dsc:.2f}",transform=axes[r,3].transAxes,ha="right",va="bottom",
                       fontsize=7.5,color="white",fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2",fc=(0,0,0,0.55),ec="none"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=TP,label="True positive"),Patch(fc=FP,label="False positive (over-seg.)"),
                        Patch(fc=FN,label="False negative (missed)")],
               loc="lower center",ncol=3,fontsize=8,bbox_to_anchor=(0.5,-0.012))
    fig.suptitle("Qualitative segmentation — representative (median-Dice) real cases",
                 fontsize=9.5,fontweight="bold",y=0.998)
    fig.tight_layout(rect=[0,0.03,1,0.985]); save(fig,"fig3_qualitative")

def fig_error():
    from scipy.ndimage import distance_transform_edt
    names=list(REPORT.keys())
    over=[1-REPORT[n][5] for n in names]   # 1 - precision  (over-segmentation)
    miss=[1-REPORT[n][3] for n in names]   # 1 - sensitivity (missed)
    # boundary-distance error profile, pooled by seen/unseen
    edges=np.arange(-25,26,2); centers=(edges[:-1]+edges[1:])/2
    acc={"seen":[np.zeros(len(centers)),np.zeros(len(centers))],
         "unseen":[np.zeros(len(centers)),np.zeros(len(centers))]}
    for name,(kind,idir,mdir) in TEST_SETS.items():
        pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(22,len(pairs))).astype(int)
        for j in idxs:
            it,mt=load_item(*pairs[j]); gt=mt[0].numpy()>0.5; pr=predict(it)>0.5
            din=distance_transform_edt(gt); dout=distance_transform_edt(~gt)
            sd=np.where(gt,-din,dout); err=(pr!=gt).astype(np.float64)
            for b in range(len(centers)):
                m=(sd>=edges[b])&(sd<edges[b+1])
                acc[kind][0][b]+=err[m].sum(); acc[kind][1][b]+=m.sum()
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.0))
    # (a) error composition
    x=np.arange(len(names)); w=0.38
    ax[0].bar(x-w/2,over,w,color=RED,   edgecolor="#333",lw=0.5,label="Over-seg. (1$-$Prec.)",zorder=3)
    ax[0].bar(x+w/2,miss,w,color=BLUE,  edgecolor="#333",lw=0.5,label="Missed (1$-$Sens.)",zorder=3)
    for xi,o,m in zip(x,over,miss):
        ax[0].text(xi-w/2,o+0.004,f"{o:.2f}",ha="center",va="bottom",fontsize=6.5)
        ax[0].text(xi+w/2,m+0.004,f"{m:.2f}",ha="center",va="bottom",fontsize=6.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[n] for n in names],rotation=22,ha="right",fontsize=7.5)
    ax[0].set_ylabel("Error rate"); ax[0].set_ylim(0,0.40); ax[0].set_title("(a) Error composition per dataset")
    ax[0].legend(loc="upper left"); ax[0].grid(axis="y",color=GRID,lw=0.6); ax[0].set_axisbelow(True); full_box(ax[0])
    # (b) error vs boundary distance
    for kind,col in [("seen",SEEN),("unseen",UNSEEN)]:
        rate=acc[kind][0]/np.maximum(acc[kind][1],1)
        ax[1].plot(centers,rate,color=col,lw=1.6,ls="-" if kind=="seen" else "--",label=kind)
    ax[1].axvline(0,color="#333",lw=0.8,ls=":")
    ax[1].text(-1.5,ax[1].get_ylim()[1]*0.99,"GT boundary",fontsize=6.8,color=INK2,ha="right",va="top")
    ax[1].set_xlabel("Signed distance from boundary (px)   [<0 inside · >0 outside]")
    ax[1].set_ylabel("Misclassification rate"); ax[1].set_title("(b) Errors concentrate at the boundary")
    ax[1].legend(loc="upper right"); ax[1].grid(color=GRID,lw=0.6); ax[1].set_axisbelow(True); full_box(ax[1])
    fig.tight_layout(); save(fig,"fig4_error_analysis")

def fig_size():
    areas=[]; dices=[]; kinds=[]
    for name,(kind,idir,mdir) in TEST_SETS.items():
        pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(110,len(pairs))).astype(int)
        for j in idxs:
            it,mt=load_item(*pairs[j]); g=mt[0].numpy()
            if (g>0.5).sum()<1: continue
            areas.append(float((g>0.5).mean())*100); dices.append(dice_of(predict(it),g)); kinds.append(kind)
    areas=np.array(areas); dices=np.array(dices); kinds=np.array(kinds)
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.0))
    # (a) scatter + binned-median trend
    for k,col in [("seen",SEEN),("unseen",UNSEEN)]:
        m=kinds==k; ax[0].scatter(areas[m],dices[m],s=9,color=col,alpha=0.30,edgecolors="none",
                                  label=f"{k} (n={m.sum()})")
    qe=np.quantile(areas,np.linspace(0,1,9)); qc=(qe[:-1]+qe[1:])/2; med=[]
    for b in range(len(qc)):
        mm=(areas>=qe[b])&(areas<=qe[b+1]); med.append(np.median(dices[mm]) if mm.sum() else np.nan)
    ax[0].plot(qc,med,color="#222",lw=1.5,marker="o",ms=3,label="median trend")
    ax[0].set_xlabel("Polyp area (% of image)"); ax[0].set_ylabel("Per-image Dice")
    ax[0].set_ylim(0,1.02); ax[0].set_xlim(0,min(40,areas.max()*1.05))
    ax[0].set_title("(a) Dice vs. lesion size"); ax[0].legend(loc="lower right",fontsize=7)
    ax[0].grid(color=GRID,lw=0.6); ax[0].set_axisbelow(True); full_box(ax[0])
    # (b) mean Dice by size class, seen vs unseen
    bins=[(areas<5),((areas>=5)&(areas<15)),(areas>=15)]; labs=["Small\n(<5%)","Medium\n(5–15%)","Large\n(>15%)"]
    x=np.arange(3); w=0.38
    for off,k,col in [(-w/2,"seen",SEEN),(w/2,"unseen",UNSEEN)]:
        vals=[]; ns=[]
        for b in bins:
            mm=b&(kinds==k); vals.append(float(dices[mm].mean()) if mm.sum() else np.nan); ns.append(int(mm.sum()))
        ax[1].bar(x+off,vals,w,color=col,edgecolor="#333",lw=0.5,zorder=3,label=k)
        for xi,v,n in zip(x,vals,ns):
            if not np.isnan(v):
                ax[1].text(xi+off,v+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=6.8,fontweight="bold")
                ax[1].text(xi+off,0.03,f"n={n}",ha="center",va="bottom",fontsize=6,color="white",rotation=90)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs,fontsize=7.5); ax[1].set_ylabel("Mean Dice")
    ax[1].set_ylim(0,1.18); ax[1].set_title("(b) Mean Dice by size class"); ax[1].legend(loc="upper right",ncol=2)
    ax[1].grid(axis="y",color=GRID,lw=0.6); ax[1].set_axisbelow(True); full_box(ax[1])
    fig.tight_layout(); save(fig,"fig5_dice_vs_size")

def fig_wavelet():
    import pywt
    pairs = list_pairs(*TEST_SETS["Kvasir-SEG"][1:])
    it, mt = load_item(*pairs[7]); img = denorm(it); gray = img.mean(-1)
    LL,(LH,HL,HH) = pywt.dwt2(gray,"haar")
    HFe = np.abs(LH)+np.abs(HL)+np.abs(HH)
    def perturb(ll,g,b):
        mu=ll.mean(); sd=ll.std()+1e-5; return ((ll-mu)/sd)*(sd*g)+(mu+b*sd)
    recon = lambda llp: np.clip(pywt.idwt2((llp,(LH,HL,HH)),"haar"),0,1)
    r0 = recon(perturb(LL,1.0,0.0)); rA = recon(perturb(LL,0.8,0.6)); rB = recon(perturb(LL,1.2,-0.6))
    vlo,vhi = r0.min(), r0.max()
    fig,ax = plt.subplots(2,5,figsize=(9.6,4.0))
    row1 = [("Input frame",img,None,None),("LL  (low-freq: style)",LL,"gray","#E8892B"),
            ("LH  (horizontal)",LH,"gray",None),("HL  (vertical)",HL,"gray",None),
            ("HH  (diagonal: edges)",HH,"gray",BLUE)]
    for a,(t,d,cm,bc) in zip(ax[0],row1):
        a.imshow(d,cmap=cm); a.set_title(t,fontsize=8.5,fontweight="bold",pad=3); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color(bc or "#cccccc"); s.set_linewidth(1.6 if bc else 0.6)
    # rA=perturb(0.8,0.6): std x0.8 (sigma down), mean +0.6*sd (mu up).  rB=perturb(1.2,-0.6): sigma up, mu down.
    row2 = [("High-freq energy",HFe,"magma"),("Recon. (baseline)",r0,"gray"),
            ("LL style A  (μ↑, σ↓)",rA,"gray"),("LL style B  (μ↓, σ↑)",rB,"gray")]
    for a,(t,d,cm) in zip(ax[1][:4],row2):
        kw = {} if cm=="magma" else dict(vmin=vlo,vmax=vhi)
        a.imshow(d,cmap=cm,**kw); a.set_title(t,fontsize=8.5,fontweight="bold",pad=3); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#cccccc"); s.set_linewidth(0.6)
    tx=ax[1][4]; tx.axis("off")
    tx.text(0.0,0.94,"Frequency-domain\nstyle perturbation",fontsize=8.7,fontweight="bold",va="top",color="#B4611A")
    tx.text(0.0,0.60,"Low frequency (LL) carries the\nscanner / illumination style;\nhigh frequency (LH·HL·HH) carries\npolyp structure & edges.",
            fontsize=7.4,va="top",color=INK)
    tx.text(0.0,0.20,"Perturbing LL in training changes\nappearance, not shape → domain-\ninvariant features.",
            fontsize=7.4,va="top",color=INK2)
    fig.suptitle("Haar wavelet decomposition and the WFM low-frequency style perturbation (real frame)",
                 fontsize=9.5,fontweight="bold",y=1.0)
    fig.tight_layout(rect=[0,0,1,0.97]); save(fig,"fig6_wavelet")

def fig_violin():
    import os as _os
    names=list(TEST_SETS.keys()); kinds=[TEST_SETS[n][0] for n in names]
    cache="outputs/figures/_violin_dice.npz"
    if _os.path.exists(cache):
        z=np.load(cache,allow_pickle=True); data=list(z["data"])
    else:
        data=[]
        for name,(kind,idir,mdir) in TEST_SETS.items():
            pairs=list_pairs(idir,mdir); idxs=np.linspace(0,len(pairs)-1,min(200,len(pairs))).astype(int)
            d=[dice_of(predict(it),mt[0].numpy()) for it,mt in (load_item(*pairs[j]) for j in idxs)]
            data.append(np.array(d))
        np.savez(cache,data=np.array(data,dtype=object))
    ns=[len(d) for d in data]
    fig,ax=plt.subplots(figsize=(7.2,3.3))
    pos=np.arange(1,len(names)+1)
    vp=ax.violinplot(data,positions=pos,widths=0.75,showextrema=False)
    for b,k in zip(vp["bodies"],kinds):
        b.set_facecolor(SEEN if k=="seen" else UNSEEN); b.set_alpha(0.45)
        b.set_edgecolor(SEEN if k=="seen" else "#B4611A"); b.set_linewidth(0.9)
    bp=ax.boxplot(data,positions=pos,widths=0.16,showfliers=False,patch_artist=True,
                  medianprops=dict(color="white",lw=1.3),
                  boxprops=dict(facecolor="#333333",edgecolor="#333333",lw=0.6),
                  whiskerprops=dict(color="#333333",lw=0.8),capprops=dict(color="#333333",lw=0.8))
    for xi,d in zip(pos,data):
        ax.scatter([xi],[d.mean()],color="white",edgecolor="#333",lw=0.6,s=16,zorder=6)
    ax.set_xticks(pos); ax.set_xticklabels([f"{SHORT[n]}\n(n={c})" for n,c in zip(names,ns)],fontsize=8)
    ax.set_ylabel("Per-image Dice"); ax.set_ylim(0,1.03); ax.set_xlim(0.4,len(names)+0.6)
    ax.set_title("Per-image Dice distribution per dataset (WaveLite-DG)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=SEEN,alpha=0.45,ec=SEEN,label="Seen"),
                       Patch(fc=UNSEEN,alpha=0.45,ec="#B4611A",label="Unseen"),
                       plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor="#333",label="mean")],
              loc="lower left",ncol=3,fontsize=7.5,handlelength=1.2)
    ax.grid(axis="y",color=GRID,lw=0.6); ax.set_axisbelow(True); full_box(ax)
    fig.tight_layout(); save(fig,"fig7_dice_distribution")

def fig_table():
    from matplotlib.patches import Rectangle
    cols=["Dataset","Images","Role in this study","Protocol"]
    rows=[("Kvasir-SEG","1,000","Train / validation / test","Seen"),
          ("CVC-ClinicDB","612","Train / validation / test","Seen"),
          ("CVC-ColonDB","380","Test only","Unseen"),
          ("ETIS-LaribPolypDB","196","Test only","Unseen"),
          ("CVC-300 (EndoScene)","60","Test only","Unseen")]
    xe=[0.015,0.345,0.505,0.815,0.985]; ncol=4
    fig,ax=plt.subplots(figsize=(7.4,2.7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.965,"Datasets and cross-centre evaluation protocol",ha="center",fontsize=10,fontweight="bold")
    y0=0.88; h=0.125
    def cell(ci,y,txt,fc,tc,bold=False,ha="center"):
        ax.add_patch(Rectangle((xe[ci],y-h),xe[ci+1]-xe[ci],h,facecolor=fc,edgecolor="#b9b9b9",lw=0.7))
        xx=(xe[ci]+xe[ci+1])/2 if ha=="center" else xe[ci]+0.012
        ax.text(xx,y-h/2,txt,ha=ha,va="center",fontsize=8.3,color=tc,fontweight="bold" if bold else "normal")
    # header
    for c in range(ncol): cell(c,y0,cols[c],"#184f95","white",bold=True)
    # body
    for r,(ds,im,role,proto) in enumerate(rows):
        yy=y0-(r+1)*h
        rowfc="#ffffff" if proto=="Seen" else "#fdf1e6"
        cell(0,yy,ds,rowfc,INK,ha="left"); cell(1,yy,im,rowfc,INK)
        cell(2,yy,role,rowfc,INK2);
        cell(3,yy,proto,rowfc,SEEN if proto=="Seen" else "#B4611A",bold=True)
    cap=("Training pool = Kvasir-SEG + CVC-ClinicDB (1,612 images), split 80/20 into "
         "1,289 training / 323 validation, at 352$\\times$352 px.\nThe three unseen datasets are held out "
         "entirely for cross-centre generalization testing (never seen during training).")
    ax.text(0.015,y0-6*h-0.03,cap,ha="left",va="top",fontsize=7.3,color=INK2)
    fig.tight_layout(); save(fig,"fig8_dataset_table")

def fig_failure():
    import os as _os
    from scipy.ndimage import binary_dilation
    unseen=[n for n in TEST_SETS if TEST_SETS[n][0]=="unseen"]
    cache="outputs/figures/_failure2.npz"
    if _os.path.exists(cache):
        z=np.load(cache,allow_pickle=True); catalog=list(map(tuple,z["cat"]))
    else:
        catalog=[]
        for name in unseen:
            _,idir,mdir=TEST_SETS[name]; pairs=list_pairs(idir,mdir)
            for ip,mp in pairs:
                it,mt=load_item(ip,mp); catalog.append((dice_of(predict(it),mt[0].numpy()),name,str(ip),str(mp)))
        np.savez(cache,cat=np.array(catalog,dtype=object))
    # worst case from EACH unseen dataset (diverse, honest)
    worst=[]
    for name in unseen:
        cand=[t for t in catalog if t[1]==name]; cand.sort(key=lambda t:float(t[0])); worst.append(cand[0])
    TP=(0.16,0.70,0.26); FP=(0.86,0.16,0.16); FN=(0.16,0.36,0.92)
    fig,axes=plt.subplots(len(worst),4,figsize=(7.4,1.9*len(worst)))
    cols=["Input","Ground truth","Prediction","Error map"]
    for r,(d,name,ip,mp) in enumerate(worst):
        it,mt=load_item(ip,mp); img=denorm(it); gt=mt[0].numpy()>0.5; pr=predict(it)>0.5
        gray=img.mean(-1); area=gt.mean()*100
        ring=binary_dilation(gt,iterations=12)&(~gt)
        contrast=abs(gray[gt].mean()-gray[ring].mean()) if gt.any() and ring.any() else 0.0
        tags=[]
        if area<1.5: tags.append("tiny lesion")
        if contrast<0.06: tags.append("low contrast")
        if not tags: tags.append("flat / ambiguous edge")
        ov=img.copy()
        for m,c in [(gt&pr,TP),((~gt)&pr,FP),(gt&(~pr),FN)]: ov[m]=0.45*ov[m]+0.55*np.array(c)
        for c,dd in enumerate([img,gt.astype(float),pr.astype(float),ov]):
            a=axes[r,c]; a.imshow(dd,cmap=None if c in (0,3) else "gray",vmin=0,vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_color("#cccccc"); s.set_linewidth(0.6)
            if r==0: a.set_title(cols[c],fontsize=8.5,fontweight="bold",pad=4)
        axes[r,0].set_ylabel(f"{SHORT[name]}\n(unseen)",fontsize=8,color="#B4611A",fontweight="bold",labelpad=3)
        axes[r,0].text(0.03,0.95,f"{', '.join(tags)}\narea {area:.1f}%  ·  Δ={contrast:.02f}",
                       transform=axes[r,0].transAxes,ha="left",va="top",fontsize=6.6,color="white",
                       bbox=dict(boxstyle="round,pad=0.2",fc=(0,0,0,0.55),ec="none"))
        axes[r,3].text(0.97,0.05,f"Dice {d:.2f}",transform=axes[r,3].transAxes,ha="right",va="bottom",
                       fontsize=7.5,color="white",fontweight="bold",bbox=dict(boxstyle="round,pad=0.2",fc=(0.7,0.1,0.1,0.8),ec="none"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=TP,label="True positive"),Patch(fc=FP,label="False positive"),
                        Patch(fc=FN,label="False negative (missed)")],
               loc="lower center",ncol=3,fontsize=8,bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("Failure cases — the model's lowest-Dice unseen predictions (honest worst cases)",
                 fontsize=9.5,fontweight="bold",y=1.0)
    fig.tight_layout(rect=[0,0.03,1,0.985]); save(fig,"fig9_failure_cases")

FIGS = {"training": fig_training, "perdataset": fig_perdataset, "qualitative": fig_qualitative,
        "error": fig_error, "size": fig_size, "wavelet": fig_wavelet, "violin": fig_violin,
        "table": fig_table, "failure": fig_failure}
if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "training"
    FIGS[name]()
