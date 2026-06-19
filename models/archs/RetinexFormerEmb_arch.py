import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2

from models.archs.RetinexFormer_arch import *


# ---------------------------------------------------------------------------
#  Chromaticity Estimator  (uniform: pools globally, expands spatially)
# ---------------------------------------------------------------------------
class Chromaticity_Estimator(nn.Module):
    """Produces two spatially-uniform feature maps (chroma_x_fea, chroma_y_fea)
    from the input image.  A small conv backbone pools globally to produce a
    per-image vector which is then broadcast across the spatial dimensions.
    This is consistent with the uniform-chromaticity assumption used in
    CHROMA-MILL-SKIP (as opposed to the spatial variant)."""

    def __init__(self, n_fea_middle):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2,
            bias=True, groups=n_fea_middle,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_x = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True)
        self.head_y = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True)

    def forward(self, img):
        # img: [B, 3, H, W]
        shared = self.depth_conv(self.conv1(img.float()))  # [B, dim, H, W]
        pooled = self.pool(shared)  # [B, dim, 1, 1]
        chroma_x = self.head_x(pooled)  # [B, dim, 1, 1]
        chroma_y = self.head_y(pooled)  # [B, dim, 1, 1]
        # Broadcast to match spatial dims of the image
        chroma_x = chroma_x.expand_as(shared)  # [B, dim, H, W]
        chroma_y = chroma_y.expand_as(shared)  # [B, dim, H, W]
        return chroma_x, chroma_y


# ---------------------------------------------------------------------------
#  IG_MSA with optional chromaticity guidance
# ---------------------------------------------------------------------------
class IG_MSA_ChromaSkip(nn.Module):
    """IG_MSA extended with two chromaticity guidance tracks.
    The three signals are fused through a learned linear gate."""

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.guidance_gate = nn.Linear(dim_head * 3, dim_head, bias=True)

    def forward(self, x_in, illu_fea_trans, chroma_x_fea_trans, chroma_y_fea_trans, I_pred):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v, illu_attn, cx_attn, cy_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp,
             illu_fea_trans.flatten(1, 2),
             chroma_x_fea_trans.flatten(1, 2),
             chroma_y_fea_trans.flatten(1, 2)),
        )

        guidance = self.guidance_gate(
            torch.cat([illu_attn, cx_attn, cy_attn], dim=-1)
        )
        v = v * guidance

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out


# ---------------------------------------------------------------------------
#  IGAB with optional chromaticity guidance
# ---------------------------------------------------------------------------
class IGAB_ChromaSkip(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA_ChromaSkip(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x, illu_fea, chroma_x_fea, chroma_y_fea, I_pred):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(
                x,
                illu_fea_trans=illu_fea.permute(0, 2, 3, 1),
                chroma_x_fea_trans=chroma_x_fea.permute(0, 2, 3, 1),
                chroma_y_fea_trans=chroma_y_fea.permute(0, 2, 3, 1),
                I_pred=I_pred,
            ) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


# ---------------------------------------------------------------------------
#  Denoiser with conditional chromaticity skip connections
# ---------------------------------------------------------------------------
class Denoiser_emb(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4],
                 use_chroma_skip=False):
        super(Denoiser_emb, self).__init__()
        self.dim = dim
        self.level = level
        self.use_chroma_skip = use_chroma_skip

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            layer_modules = [
                (IGAB_ChromaSkip if use_chroma_skip else IGAB)(
                    dim=dim_level, num_blocks=num_blocks[i],
                    dim_head=dim, heads=dim_level // dim,
                ),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # FeaDownSample
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # IlluFeaDown
            ]
            if use_chroma_skip:
                layer_modules.append(nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False))  # ChromaXDown
                layer_modules.append(nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False))  # ChromaYDown
            self.encoder_layers.append(nn.ModuleList(layer_modules))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = (IGAB_ChromaSkip if use_chroma_skip else IGAB)(
            dim=dim_level, dim_head=dim,
            heads=dim_level // dim, num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                (IGAB_ChromaSkip if use_chroma_skip else IGAB)(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                    dim_head=dim, heads=(dim_level // 2) // dim,
                ),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

        self.embedding_done = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea, I, I_pred=None,
                chroma_x_fea=None, chroma_y_fea=None):
        """
        x:          [b,c,h,w]         x is feature, not image
        illu_fea:   [b,c,h,w]
        chroma_x_fea: [b,c,h,w]  (only when use_chroma_skip=True)
        chroma_y_fea: [b,c,h,w]  (only when use_chroma_skip=True)
        return out: [b,c,h,w]
        """
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        chroma_x_fea_list = []
        chroma_y_fea_list = []

        if self.use_chroma_skip:
            for (igab, FeaDown, IlluDown, ChromaXDown, ChromaYDown) in self.encoder_layers:
                fea = igab(fea, illu_fea, chroma_x_fea, chroma_y_fea, I_pred)
                illu_fea_list.append(illu_fea)
                chroma_x_fea_list.append(chroma_x_fea)
                chroma_y_fea_list.append(chroma_y_fea)
                fea_encoder.append(fea)
                fea = FeaDown(fea)
                illu_fea = IlluDown(illu_fea)
                chroma_x_fea = ChromaXDown(chroma_x_fea)
                chroma_y_fea = ChromaYDown(chroma_y_fea)

            fea = self.bottleneck(fea, illu_fea, chroma_x_fea, chroma_y_fea, I_pred)
        else:
            for (IGAB_block, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
                fea = IGAB_block(fea, illu_fea, I_pred)
                illu_fea_list.append(illu_fea)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)
                illu_fea = IlluFeaDownsample(illu_fea)

            fea = self.bottleneck(fea, illu_fea, I_pred)

        try:
            illu_pred = fea[:, 0:1, :, :].clone()
            chroma_pred = fea[:, 1:3, :, :].clone()
            self.embedding_done = fea[:, 3:, :, :].clone()
        except:
            self.embedding_done = fea.clone()

        # Decoder
        if self.use_chroma_skip:
            for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
                fea = FeaUpSample(fea)
                fea = Fusion(
                    torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
                illu_fea = illu_fea_list[self.level - 1 - i]
                chroma_x_fea = chroma_x_fea_list[self.level - 1 - i]
                chroma_y_fea = chroma_y_fea_list[self.level - 1 - i]
                fea = LeWinBlock(fea, illu_fea, chroma_x_fea, chroma_y_fea, I_pred)
        else:
            for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
                fea = FeaUpSample(fea)
                fea = Fution(
                    torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
                illu_fea = illu_fea_list[self.level - 1 - i]
                fea = LeWinBlcok(fea, illu_fea, I_pred)

        # Mapping
        out = self.mapping(fea) + x

        return out, illu_pred, chroma_pred

    def getEmbedding(self):
        return self.embedding_done

class RetinexFormer_Single_Stage_Emb(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2,
                 num_blocks=[1, 1, 1], use_prior=True, I=0, use_I=False,
                 use_he=False, use_chroma_skip=False):
        super(RetinexFormer_Single_Stage_Emb, self).__init__()
        self.use_chroma_skip = use_chroma_skip
        self.estimator = Illumination_Estimator(n_feat, use_prior=use_prior, I=I, use_I=use_I, use_he=use_he)
        if use_chroma_skip:
            self.chroma_estimator = Chromaticity_Estimator(n_feat)
        self.denoiser = Denoiser_emb(
            in_dim=in_channels, out_dim=out_channels, dim=n_feat,
            level=level, num_blocks=num_blocks, use_chroma_skip=use_chroma_skip,
        )

    def forward(self, img):
        if type(img) is tuple:
            img, I = img
        else:
            I = None

        illu_fea, illu_map, intensity = self.estimator(img, I=I)

        chroma_x_fea, chroma_y_fea = None, None
        if self.use_chroma_skip:
            chroma_x_fea, chroma_y_fea = self.chroma_estimator(img)

        input_img = img
        output_img, illu_pred, chroma_pred = self.denoiser(
            input_img, illu_fea, I=I, I_pred=intensity,
            chroma_x_fea=chroma_x_fea, chroma_y_fea=chroma_y_fea,
        )

        embedding = self.denoiser.getEmbedding()

        return output_img, illu_pred, chroma_pred, embedding


class RetinexFormerEmb(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3,
                 num_blocks=[1, 1, 1], use_prior=True, I=0, use_I=False,
                 use_he=False, use_chroma_skip=False):
        super(RetinexFormerEmb, self).__init__()
        self.stage = stage

        modules_body = [
            RetinexFormer_Single_Stage_Emb(
                in_channels=in_channels, out_channels=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks,
                use_prior=use_prior, I=I, use_I=use_I, use_he=use_he,
                use_chroma_skip=use_chroma_skip,
            )
            for _ in range(stage)
        ]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, I=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out, illu_pred, chroma_pred, embedding = self.body((x, I))

        return out, illu_pred, chroma_pred, embedding


# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = RetinexFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')
