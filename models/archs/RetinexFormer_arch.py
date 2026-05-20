import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3, use_prior = True, I = 0, use_I = False, use_he = False):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.use_I = use_I 
        self.use_prior = use_prior
        self.use_he = use_he

        if self.use_I and self.use_prior:
            n_fea_in += 1
        
        if self.use_he:
            n_fea_in += 3

        # self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(3, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        # self.conv2 = nn.Conv2d(n_fea_middle, n_fea_middle*2, kernel_size=1, bias=True)
        # self.conv3 = nn.Conv2d(n_fea_middle*2, n_fea_middle*4, kernel_size=5, stride=2, padding=2, bias=True)
        # self.conv4 = nn.Conv2d(n_fea_middle*4, n_fea_middle*4, kernel_size=5, stride=2, padding=2, bias=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(n_fea_middle*2, 1, bias=True)
        
        self.c1 = nn.Conv2d(3, 10, kernel_size=7, stride=3, padding=0, bias=False)
        self.c2 = nn.Conv2d(10, 25, kernel_size=7, stride=3, padding=0, bias=False)
        self.c3 = nn.Conv2d(25, 50, kernel_size=7, stride=2, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(50, 1, bias=True)

    def forward(self, img, I = None):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        # if self.use_prior:
        #     mean_c = img.mean(dim=1).unsqueeze(1)
        #     if self.use_I:
        #         I = I.to(img.device)  # Move I tensor to the same device as img
        #         I_c = torch.ones(I.shape[0], 1, img.shape[2], img.shape[3]).to(img.device) * I[:, None, None, None]
        #         mean_c = torch.cat([mean_c, I_c], dim=1)
        # else:
        #     if self.use_I:
        #         I = I.to(img.device)  # Move I tensor to the same device as img
        #         mean_c = torch.ones(I.shape[0], 1, img.shape[2], img.shape[3]).to(img.device) * I[:, None, None, None]
        #     else:
        #         mean_c = torch.zeros_like(img[:, 0:1, :, :]).to(img.device)
        
        # input = torch.cat([img, mean_c], dim=1)
        input = img

        # if self.use_he: 
        #     img_enhanced = adjust_gamma(img, 1/3)

        #     input = torch.cat([input, img_enhanced], dim=1)


        x_1 = self.conv1(input.float())
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        # x_ = self.conv3(illu_map)
        # x_ = self.conv4(x_)
        # x_ = self.avgpool(illu_map)
        # x_ = x_.view(x_.size(0), -1)
        # intensity = self.linear(x_)
        # intensity = intensity.squeeze()
        intensity = self.c1(img)
        intensity = self.c2(intensity)
        intensity = self.c3(intensity)
        intensity = self.avgpool(intensity)
        intensity = intensity.view(intensity.size(0), -1)
        intensity = self.linear(intensity)
        intensity = intensity.squeeze()

        return illu_fea, illu_map, intensity



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
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

    def forward(self, x_in, illu_fea_trans, I_pred):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn #  * I_pred.reshape(b, 1, 1, 1)
        # print(v.shape)
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea, I_pred):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1), I_pred=I_pred) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, kernel_size = 3, stride = 1, padding = 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                    IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                    # nn.Conv2d(in_dim, out_dim, kernel, stride, padd)
                    nn.Conv2d(dim_level, dim_level * 2, kernel_size = 4, stride = 2, padding = 1, bias=False),
                    nn.Conv2d(dim_level, dim_level * 2, kernel_size = 4, stride = 2, padding = 1, bias=False)
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea, I, I_pred): # x: input image    illu_fea: output of illumination estimator I: real intensity
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
        
        # print("fea.shape:", fea.shape)

        # try:
            # OUR ADDITION: change the illu_fea last channel for a channel of ones * lamda
        illu_pred = fea[:,0:1,:,:].clone()
        # fea[:,0:1,:,:] = torch.ones_like(fea[:,0:1,:,:]) * I.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(fea.device)
        # except:
        #     pass

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea_encoder_level = fea_encoder[self.level - 1 - i]
            # Adjust the size of fea_encoder_level to match fea
            fea_encoder_level = fea_encoder_level[:, :, :fea.shape[2], :fea.shape[3]]
            fea = Fution(torch.cat([fea, fea_encoder_level], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea, illu_fea, I_pred)

        # Mapping
        out = self.mapping(fea) + x

        return out, illu_pred


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1], use_prior=True, I = 0, use_I = False, use_he=False):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat, use_prior = use_prior, I = I, use_I = use_I, use_he = use_he)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
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
        output_img, illu_pred = self.denoiser(input_img, illu_fea, I = I, I_pred = intensity)

        return output_img, intensity.squeeze(), illu_map, input_img


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1], use_prior=True, I = 0, use_I = False, use_he=False):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks, use_prior=use_prior, I = I, use_I = use_I, use_he=use_he)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x, I = None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out, illu_pred, illu_map, input_img = self.body((x, I))

        return out, illu_pred, illu_map, input_img


# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = RetinexFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')