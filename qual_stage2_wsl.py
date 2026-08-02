# Stage 2 (run in WSL, mamba-ssm GPU): VM-UNet preds on the SAME selected images as stage 1.
import sys, json, os
sys.path.insert(0, "/mnt/d/AI-Projects/WaveMamba/_baselines/VM-UNet")
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image
from models.vmunet.vmunet import VMUNet

ROOT="/mnt/d/AI-Projects/WaveMamba"; DEVICE=torch.device("cuda")
IMG=352; MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
def win2wsl(p):
    p=p.replace("\\","/")
    if len(p)>2 and p[1]==":": p="/mnt/"+p[0].lower()+p[2:]
    return p
sel=json.load(open(f"{ROOT}/outputs/qual_selection.json"))

def load_item(ip,mp):
    img=Image.open(ip).convert("RGB").resize((IMG,IMG),Image.BILINEAR)
    msk=Image.open(mp).convert("L").resize((IMG,IMG),Image.NEAREST)
    return TF.normalize(TF.to_tensor(img),MEAN,STD),(TF.to_tensor(msk)>0.5).float()
def logit_of(model,x): return model.vmunet(x)
@torch.no_grad()
def tta_prob(model,it):
    x=it.unsqueeze(0).to(DEVICE); outs=[]
    for f in [lambda t:t,lambda t:torch.flip(t,[-1]),lambda t:torch.flip(t,[-2])]:
        outs.append(torch.sigmoid(f(logit_of(model,f(x)))))
    return torch.stack(outs).mean(0)[0,0].cpu().numpy()

model=VMUNet(num_classes=1,input_channels=3,depths=[2,2,9,2],depths_decoder=[2,9,2,2],
             drop_path_rate=0.2,load_ckpt_path=f"{ROOT}/_baselines/VM-UNet/pretrained_weights/vmamba_tiny_e292.pth").to(DEVICE)
model.load_from()
model.load_state_dict(torch.load(f"{ROOT}/outputs/baselines/vmunet_best.pt",map_location=DEVICE,weights_only=True))
model.eval()

preds=[]; dice=[]
for s in sel:
    it,mt=load_item(win2wsl(s["img"]),win2wsl(s["mask"])); pr=tta_prob(model,it)
    pb=(pr>0.5).astype(np.uint8); g=(mt[0].numpy()>0.5).astype(np.float64)
    inter=(pb*g).sum(); dc=(2*inter+1)/(pb.sum()+g.sum()+1)
    preds.append(pb); dice.append(float(dc))
    print(f"{s['dataset']:20s} VM-UNet Dice {dc:.3f}",flush=True)
np.savez_compressed(f"{ROOT}/outputs/qual_vmunet.npz",pred_vmunet=np.stack(preds))
json.dump(dice,open(f"{ROOT}/outputs/qual_dice_vmunet.json","w"))
print("STAGE2_DONE",flush=True)
