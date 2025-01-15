import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dconv_model import FusedPooling, LayerNorm2d, SCAM
from unet import UNetCompress, UNetDecompress
from model_convnext import knowledge_adaptation_convnext


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def dwt_haar(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

    def forward(self, x):
        return self.dwt_haar(x)


class DWT_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()

        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CAM(nn.Module):
    def __init__(self, num_channels, compress_factor=8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // compress_factor, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_channels // compress_factor, num_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.model(y)
        return x * y


class DynamicDepthwiseConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, num_cnvs=4, K=2, device='cuda'):
        super(DynamicDepthwiseConvolution, self).__init__()

        self.convs = nn.ModuleList([
                        nn.Conv2d(in_ch, in_ch * K, k, stride, groups=in_ch, padding='same', padding_mode='reflect')
                            for _ in range(num_cnvs)])

        self.weights = nn.Parameter(1 / num_cnvs * torch.ones((num_cnvs, 1), device=device, dtype=torch.float),
                                    requires_grad=True)
        self.final_conv = nn.Conv2d(in_channels=in_ch * K, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        feats = 0
        for i, conv in enumerate(self.convs):
            feats += self.weights[i] * conv(x)

        return self.final_conv(feats)


class SimplifiedDepthwiseCA(nn.Module):
    def __init__(self, num_channels, k, K, device="cuda"):
        super().__init__()
        self.attention = CAM(num_channels)
        self.dc = DynamicDepthwiseConvolution(in_ch=num_channels, out_ch=num_channels, K=K, k=k, device=device)

    def forward(self, x):
        q = self.dc(x)
        w = self.attention(q)
        return torch.sigmoid(w * q)


class BlockRGB(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz=3, dropout_prob=0.5,  device="cuda"):
        super(BlockRGB, self).__init__()
        self.ln = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op1 = nn.LeakyReLU(0.2)
        self.dyndc = SimplifiedDepthwiseCA(num_channels=out_ch // 2, k=13, K=4, device=device)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op2 = nn.LeakyReLU(0.2)

        self.rconv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 2, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        self.rconv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        
        self.a1 = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32), requires_grad=True)
        self.dropout1 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()

    def forward(self, x):
        xf = self.ln(x)
        xf = self.op1(self.conv1(xf))
        xf = self.dropout1(xf)
        xf += self.a1 * self.rconv1(x)
        xf = self.dyndc(xf)
        xf = self.dropout2(xf)
        xf = self.op2(self.conv2(xf))
        return xf + self.a2 * self.rconv2(x)


class IFBlendDown(nn.Module):
    def __init__(self, in_size, rgb_in_size, out_size, dwt_size=1, dropout=0.0, default=False, device="cuda", blend=False):
        super().__init__()

        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_dize = dwt_size
        self.rgb_in_size = rgb_in_size

        if dwt_size > 0:
            self.dwt = DWT_block(in_channels=in_size, out_channels=dwt_size)

        self.b_unet = UNetCompress(in_size, out_size, dropout=dropout/2)

        if default:
            self.rgb_block = BlockRGB(3, out_size, dropout_prob=dropout, device=device)
        else:
            self.rgb_block = BlockRGB(rgb_in_size, out_size, dropout_prob=dropout, device=device)

        self.fp = FusedPooling(nc=out_size, blend=blend)

    def forward(self, x, rgb_img):
        xu = self.b_unet(x)
        b, c, h, w = xu.shape
        rgb_feats = self.fp(self.rgb_block(rgb_img))

        if self.dwt_dize > 0:
            lfw, hfw = self.dwt(x)
            return torch.cat((xu, rgb_feats[:, :c, :, :], lfw), dim=1), hfw, xu, rgb_feats[:, c:, :, :]
        else:
            return torch.cat((xu, rgb_feats[:, :c, :, :]), dim=1), None, xu, rgb_feats[:, c:, :, :]


class WASAM(nn.Module):
    '''
    Based on NAFNET Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c_rgb, cr):
        super().__init__()
        self.scale = (0.5 * (c_rgb + cr)) ** -0.5

        self.norm_l = LayerNorm2d(c_rgb)
        self.l_proj_res = nn.Conv2d(c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj_res = nn.Conv2d(cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)

        self.l_proj1 = nn.Conv2d(c_rgb, c_rgb, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(cr, c_rgb, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c_rgb // 2, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_rgb // 2, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x_rgb, x_hfw):
        Q_l = self.l_proj1(self.norm_l(x_rgb)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(x_hfw).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_rgb).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_hfw).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return torch.cat((self.l_proj_res(x_rgb) + F_r2l, self.r_proj_res(x_hfw) + F_l2r), dim=1)


class IFBlendUp(nn.Module):
    def __init__(self, in_size, rgb_size, dwt_size,  out_size, dropout):
        super().__init__()
        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_size = dwt_size
        self.rgb_size = rgb_size

        self.b_unet = UNetDecompress(in_size + dwt_size, out_size, dropout=dropout)
        self.rgb_proj = nn.ConvTranspose2d(in_channels=rgb_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)
        if dwt_size > 0:
           self.spfam = WASAM(rgb_size, dwt_size)

    def forward(self, x, hfw, rgb):
        if self.dwt_size > 0:
            rgb = self.spfam(rgb, hfw)

        state = self.b_unet(torch.cat((x, hfw), dim=1))
        state = state + F.relu(self.rgb_proj(rgb))
        return state


class IFBlend(nn.Module):
    def __init__(self, in_channels, device="cuda", use_gcb=False, blend=False):
        super().__init__()

        self.in_channels = in_channels
        self.use_gcb = use_gcb

        self.in_conv = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.in_bn = nn.BatchNorm2d(in_channels)

        if self.use_gcb:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels + 28, 3, kernel_size=7, padding=3, padding_mode="reflect")
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=7, padding=3, padding_mode="reflect")
            )

        if self.use_gcb:
            self.gcb = knowledge_adaptation_convnext()

        self.c1 = IFBlendDown(in_size=in_channels, rgb_in_size=3,
                              out_size=32, dwt_size=1, dropout=0.15, default=True, device=device, blend=blend)
        self.c2 = IFBlendDown(in_size=65, rgb_in_size=32, out_size=64, dwt_size=2, dropout=0.2, device=device, blend=blend)
        self.c3 = IFBlendDown(in_size=130, rgb_in_size=64, out_size=128, dwt_size=4, dropout=0.25, device=device, blend=blend)
        self.c4 = IFBlendDown(in_size=260, rgb_in_size=128, out_size=256, dwt_size=8, dropout=0.3, device=device, blend=blend)
        self.c5 = IFBlendDown(in_size=520, rgb_in_size=256, out_size=256, dwt_size=16, dropout=0.0, device=device, blend=blend)

        self.d5 = IFBlendUp(in_size=528, dwt_size=16, rgb_size=256, out_size=256, dropout=0.0)
        self.d4 = IFBlendUp(in_size=512, dwt_size=8, rgb_size=256, out_size=128, dropout=0.3)
        self.d3 = IFBlendUp(in_size=256, dwt_size=4, rgb_size=128, out_size=64, dropout=0.25)
        self.d2 = IFBlendUp(in_size=128, dwt_size=2, rgb_size=64, out_size=32, dropout=0.2)
        self.d1 = IFBlendUp(in_size=64, dwt_size=1, rgb_size=32, out_size=in_channels, dropout=0.1)

    def forward(self, x):
        x_rgb = x
        xf = self.in_bn(self.in_conv(x))
        x1, s1, xs1, rgb1 = self.c1(xf, x_rgb)
        x2, s2, xs2,  rgb2 = self.c2(x1, rgb1)
        x3, s3, xs3, rgb3 = self.c3(x2, rgb2)
        x4, s4, xs4, rgb4 = self.c4(x3, rgb3)
        x5, s5, xs5, rgb5 = self.c5(x4, rgb4)
        y5 = self.d5(x5, s5, rgb5)
        y4 = self.d4(torch.cat((y5, xs4), dim=1), s4, rgb4)
        y3 = self.d3(torch.cat((y4, xs3), dim=1), s3, rgb3)
        y2 = self.d2(torch.cat((y3, xs2), dim=1), s2, rgb2)
        y1 = self.d1(torch.cat((y2, xs1), dim=1), s1, rgb1)

        if self.use_gcb:
            return torch.sigmoid(x + self.out_conv(torch.cat((y1, self.gcb(x_rgb)), dim=1)))
        else:
            return torch.sigmoid(x + self.out_conv(y1))


if __name__ == '__main__':
    inp = torch.rand((1, 3, 512, 512))
    model = IFBlend(16, device="cpu")
    out = model(inp)
    print(out.shape)
