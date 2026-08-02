# Faithful SSFormer-S: timm pretrained PVTv2-B2 encoder + SSFormer's real PLD Decoder.
# Decoder/conv/resize copied verbatim from _baselines/ssformer/models/pvt/pvt_PLD.py;
# ConvModule is a shim of mmcv.cnn.ConvModule (conv + BN + ReLU, its documented default).
import torch, torch.nn as nn, torch.nn.functional as F

def resize(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return F.interpolate(x, size, scale_factor, mode, align_corners)

class ConvModule(nn.Module):   # mmcv.cnn.ConvModule shim: Conv(bias=not norm) -> Norm -> Act(ReLU)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              bias=(norm_cfg is None))
        self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
        self.act = nn.ReLU(inplace=True) if act_cfg is not None else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.act is not None: x = self.act(x)
        return x

class _conv(nn.Module):   # SSFormer's `conv` linear-embedding (verbatim)
    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Decoder(nn.Module):   # SSFormer Progressive Locality Decoder (verbatim)
    def __init__(self, dims, dim, class_num=1):
        super().__init__()
        c1, c2, c3, c4 = dims; ed = dim
        self.linear_c4 = _conv(c4, ed); self.linear_c3 = _conv(c3, ed)
        self.linear_c2 = _conv(c2, ed); self.linear_c1 = _conv(c1, ed)
        self.linear_fuse34 = ConvModule(ed*2, ed, 1, norm_cfg=dict(type='BN'))
        self.linear_fuse2  = ConvModule(ed*2, ed, 1, norm_cfg=dict(type='BN'))
        self.linear_fuse1  = ConvModule(ed*2, ed, 1, norm_cfg=dict(type='BN'))
        self.linear_pred = nn.Conv2d(ed, class_num, 1); self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs; n = c4.shape[0]
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n,-1,c4.shape[2],c4.shape[3]); _c4 = resize(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3]); _c3 = resize(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3]); _c2 = resize(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n,-1,c1.shape[2],c1.shape[3])
        L34 = self.linear_fuse34(torch.cat([_c4,_c3],1))
        L2  = self.linear_fuse2(torch.cat([L34,_c2],1))
        _c  = self.linear_fuse1(torch.cat([L2,_c1],1))
        return self.linear_pred(self.dropout(_c))

class SSFormerS(nn.Module):   # SSFormer-S = pretrained PVTv2-B2 + PLD decoder
    def __init__(self, class_num=1):
        super().__init__()
        import timm
        self.backbone = timm.create_model("pvt_v2_b2", features_only=True, pretrained=True)
        ch = self.backbone.feature_info.channels()   # [64,128,320,512]
        self.decode_head = Decoder(dims=ch, dim=256, class_num=class_num)
    def forward(self, x):
        feats = self.backbone(x)                      # 4 maps, strides 4/8/16/32
        out = self.decode_head(feats)                 # at 1/4 res
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return (out,)                                 # 1-tuple -> harness applies structure_loss
