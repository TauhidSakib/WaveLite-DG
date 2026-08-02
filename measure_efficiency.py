# Measured efficiency for all Windows-side models: params, GFLOPs (thop, approx), single-pass & 3-view-TTA
# latency/FPS, peak GPU memory. Same GPU, batch 1, 352x352, FP32, warmed up, CUDA-synchronized. VM-UNet measured in WSL.
import os, sys, json, time
os.chdir(r"D:\AI-Projects\WaveMamba")
import torch
from figlib import WaveMambaDG, DEVICE

def build(arch):
    if arch=="ours": return WaveMambaDG(pretrained=False)
    if arch=="unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    if arch=="pranet":
        sys.path.insert(0,"_baselines/PraNet"); from lib.PraNet_Res2Net import PraNet
        return PraNet()
    if arch=="polyppvt":
        sys.path.insert(0,"_baselines/Polyp-PVT"); from lib.pvt import PolypPVT; return PolypPVT()
    if arch=="ssformer":
        from ssformer_pld import SSFormerS; return SSFormerS(1)

@torch.no_grad()
def bench(model, tta=False, iters=40, warm=10):
    model.eval().to(DEVICE)
    x=torch.randn(1,3,352,352,device=DEVICE)
    def once():
        if tta:
            for f in [lambda t:t, lambda t:torch.flip(t,[-1]), lambda t:torch.flip(t,[-2])]:
                model(f(x))
        else:
            model(x)
    for _ in range(warm): once()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t=time.time()
    for _ in range(iters): once()
    torch.cuda.synchronize(); dt=(time.time()-t)/iters
    mem=torch.cuda.max_memory_allocated()/1024/1024
    return 1.0/dt, dt*1000, mem

res={}
for arch in ["ours","unet","pranet","polyppvt","ssformer"]:
    try:
        m=build(arch)
        p=sum(x.numel() for x in m.parameters())/1e6
        gflops=None
        try:
            from thop import profile
            macs,_=profile(build(arch).to(DEVICE), inputs=(torch.randn(1,3,352,352,device=DEVICE),), verbose=False)
            gflops=round(macs/1e9,2)
        except Exception as e:
            gflops=f"n/a ({type(e).__name__})"
        fps1,lat1,mem1=bench(m,tta=False)
        fpst,latt,memt=bench(m,tta=True)
        res[arch]={"params_M":round(p,2),"gflops":gflops,
                   "fps_single":round(fps1,1),"lat_ms_single":round(lat1,1),
                   "fps_tta":round(fpst,1),"lat_ms_tta":round(latt,1),
                   "peak_MB":round(max(mem1,memt))}
        print(arch, res[arch], flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc(); res[arch]={"error":str(e)}
json.dump(res, open("outputs/efficiency.json","w"), indent=1)
print("EFF_DONE", flush=True)
