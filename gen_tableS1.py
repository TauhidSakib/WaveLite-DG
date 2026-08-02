# Supplementary Table S1: full metrics (Dice, IoU, Sensitivity, Specificity, Precision) for all models/datasets.
import os, json
os.chdir(r"D:\AI-Projects\WaveMamba")
OUTB="outputs/baselines"
MODELS=[("WaveLite-DG","ours"),("U-Net","unet"),("PraNet","pranet"),
        ("Polyp-PVT","polyppvt"),("SSFormer-S","ssformer"),("VM-UNet","vmunet")]
DS=[("Kvasir-SEG","Kvasir*"),("CVC-ClinicDB","ClinicDB*"),("CVC-ColonDB","ColonDB"),
    ("ETIS-LaribPolypDB","ETIS"),("CVC-300","CVC-300")]
METRICS=[("Dice",1),("IoU",2),("Sensitivity",3),("Specificity",4),("Precision",5)]
rep={m[1]:json.load(open(f"{OUTB}/{m[1]}_report.json"))["report"] for m in MODELS}

lines=["# Supplementary Table S1 — Full per-dataset metrics (fair held-out protocol, TTA)","",
       "*Seen columns (\\*) are validation results. All values are our reproductions on the fixed cross-dataset split.*",""]
for mname,mi in METRICS:
    lines.append(f"## S1.{mi} {mname}")
    header="| Model | "+" | ".join(d[1] for d in DS)+" |"
    sep="|---|"+"---|"*len(DS)
    lines+=[header,sep]
    for disp,key in MODELS:
        row=[disp]+[f"{rep[key][d[0]][mi]:.4f}" for d in DS]
        lines.append("| "+" | ".join(row)+" |")
    lines.append("")
open("outputs/supplementary_tableS1.md","w",encoding="utf-8").write("\n".join(lines))
print("wrote outputs/supplementary_tableS1.md")
print("\n".join(lines[:14]))
