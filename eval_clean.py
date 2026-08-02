# Clean re-evaluation (TTA), matching the reported protocol, after quarantining junk file 380.
import numpy as np, json, torch
from figlib import *

@torch.no_grad()
def tta_prob(it):
    x=it.unsqueeze(0).to(DEVICE); m=get_model()
    outs=[]
    for f in [lambda t:t, lambda t:torch.flip(t,[-1]), lambda t:torch.flip(t,[-2])]:
        o=m(f(x))["mask_logit"]; outs.append(torch.sigmoid(f(o)))
    return torch.stack(outs).mean(0)[0,0].cpu().numpy()

def metrics(pr,gt,thr=0.5):
    p=(pr>thr).astype(np.float64); t=(gt>0.5).astype(np.float64)
    tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum(); tn=((1-p)*(1-t)).sum()
    return (2*tp/(2*tp+fp+fn+1e-8), tp/(tp+fp+fn+1e-8), tp/(tp+fn+1e-8), tn/(tn+fp+1e-8), tp/(tp+fp+1e-8))

rows={}
for name,(kind,idir,mdir) in TEST_SETS.items():
    pairs=list_pairs(idir,mdir); acc=np.zeros(5); n=0
    for ip,mp in pairs:
        it,mt=load_item(ip,mp); pr=tta_prob(it); gt=mt[0].numpy()
        acc+=np.array(metrics(pr,gt)); n+=1
    acc/=n; rows[name]=(kind,*[round(float(v),4) for v in acc],n)
    print(f"{name:20s} {kind:6s} Dice {acc[0]:.4f} IoU {acc[1]:.4f} Sens {acc[2]:.4f} Spec {acc[3]:.4f} Prec {acc[4]:.4f} n={n}",flush=True)

import csv
with open("outputs/segmentation_report_clean.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","kind","dice","iou","sensitivity","specificity","precision","n"])
    for name,(kind,d,i,se,sp,pr,n) in rows.items(): w.writerow([name,kind,d,i,se,sp,pr,n])
json.dump({k:list(v) for k,v in rows.items()}, open("outputs/report_clean.json","w"), indent=1)
print("\nCLEAN_EVAL_DONE -> outputs/segmentation_report_clean.csv")
