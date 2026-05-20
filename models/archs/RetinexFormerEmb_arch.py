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


class Denoiser_emb(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser_emb, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                    IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                    # nn.Conv2d(in_dim, out_dim, kernel, stride, padd)
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        # print("dim:", dim_level, "dim_head:", dim, "heads:", dim_level // dim, "num_blocks:", num_blocks[-1])
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])


        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            # if i == 0:
            #     dim_level = dim_level + 1
            
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim),
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

    def forward(self, x, illu_fea, I, I_pred=None): # x: input image    illu_fea: output of illumination estimator I: real intensity
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # print(x.shape, illu_fea.shape, I.shape)
        # Embedding
        fea = self.embedding(x) # light-up feature

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea, I_pred)  # bchw
            illu_fea_list.append(illu_fea)      # save F_x after the attention module
            fea_encoder.append(fea)             # save fea in each level (it will have different sizes)
            fea = FeaDownSample(fea)            # Convolution
            illu_fea = IlluFeaDownsample(illu_fea) # Convolution

        # print("illu_pred.shape:", illu_pred.shape, "I.shape:", I)
        # print(I.unsqueeze(1).unsqueeze(1).unsqueeze(1).shape)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea, I_pred)

        try:
            # OUR ADDITION: change the illu_fea last channel for a channel of ones * lamda
            illu_pred = fea[:,0:1,:,:].clone()
            # fea[:,0:1,:,:] = torch.ones_like(fea[:,0:1,:,:]) * I.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(fea.device)
            chroma_pred = fea[:, 1:3, :, :].clone() # CHROMA-MILL
            self.embedding_done = fea[:, 3:, :, :].clone()
        except:
            self.embedding_done = fea.clone()

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea, illu_fea, I_pred)

        # Mapping
        out = self.mapping(fea) + x

        return out, illu_pred, chroma_pred

    def getEmbedding(self):
        return self.embedding_done

class RetinexFormer_Single_Stage_Emb(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1], use_prior=True, I = 0, use_I = False, use_he = False):
        super(RetinexFormer_Single_Stage_Emb, self).__init__()
        self.estimator = Illumination_Estimator(n_feat, use_prior = use_prior, I = I, use_I = use_I, use_he = use_he)
        self.denoiser = Denoiser_emb(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        if type(img) is tuple:
            img, I = img
        else:
            I = None

        # print("img.shape:", img.shape, "I.shape:", I)
        illu_fea, illu_map, intensity = self.estimator(img, I = I)
        input_img = img # * illu_map + img
        output_img, illu_pred, chroma_pred = self.denoiser(input_img, illu_fea, I = I, I_pred = intensity)

        embedding = self.denoiser.getEmbedding()

        return output_img, illu_pred, chroma_pred, embedding


class RetinexFormerEmb(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1], use_prior=True, I = 0, use_I = False, use_he = False):
        super(RetinexFormerEmb, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage_Emb(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks, use_prior=use_prior, I = I, use_I = use_I, use_he = use_he)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x, I = None):
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