import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from kornia.geometry.transform import rotate
import pdb


class DualFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1,spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(DualFourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer_vert = weight_norm(torch.nn.Conv2d(in_channels=in_channels  + (1 if spectral_pos_encoding else 0),
                                          out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.conv_layer_hor = weight_norm(torch.nn.Conv2d(in_channels=in_channels + (1 if spectral_pos_encoding else 0),
                                          out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.relu  = torch.nn.LeakyReLU(negative_slope=0.05, inplace=True)
        
        self.spectral_pos_encoding = spectral_pos_encoding
        self.fft_norm = fft_norm

    def forward(self, x):
        batch,c,height,weight = x.shape # (batch, c, h, w)

        horizontal_x, vertical_x= torch.chunk(x,2,dim=1)

        horizontal_ffted = torch.fft.rfft(horizontal_x, dim=(-1), norm=self.fft_norm) # (batch, c/2, h, w/2+1) 
        horizontal_ffted = torch.cat((horizontal_ffted.real, horizontal_ffted.imag), dim=1)        # (batch, c, h, w/2+1)

        vertical_ffted = torch.fft.rfft(vertical_x, dim=(-2), norm=self.fft_norm) # (batch, c/2, h/2+1, w) 
        vertical_ffted = torch.cat((vertical_ffted.real, vertical_ffted.imag), dim=1)        # (batch, c, h/2+1, w)

        if self.spectral_pos_encoding:
            horizontal_height, horizontal_width = horizontal_ffted.shape[-2:]
            vertical_height, vertical_width = vertical_ffted.shape[-2:]
            coords_hor = torch.linspace(0, 1, horizontal_width)[None, None, None, :].expand(batch, 1, horizontal_height, horizontal_width).to(horizontal_ffted)
            coords_vert = torch.linspace(0, 1, vertical_height)[None, None, :, None].expand(batch, 1, vertical_height, vertical_width).to(vertical_ffted)
            horizontal_ffted = torch.cat((coords_hor, horizontal_ffted), dim=1)
            vertical_ffted = torch.cat((coords_vert, vertical_ffted), dim=1)

        horizontal_ffted = self.conv_layer_hor(horizontal_ffted)  # (batch, c, h, w/2+1)
        horizontal_ffted = self.relu(horizontal_ffted)

        vertical_ffted = self.conv_layer_vert(vertical_ffted)  # (batch, c, h/2+1, w)
        vertical_ffted = self.relu(vertical_ffted)

        horizontal_ffted_real, horizontal_ffted_imag= torch.chunk(horizontal_ffted,2,dim=1)
        horizontal_ffted = torch.complex(horizontal_ffted_real, horizontal_ffted_imag) # (batch, c/2, h, w/2+1)
        horizontal_output = torch.fft.irfft(horizontal_ffted, n=(weight), dim=(-1), norm=self.fft_norm)# (batch, c/2, h, w)

        vertical_ffted_real, vertical_ffted_imag= torch.chunk(vertical_ffted,2,dim=1)
        vertical_ffted = torch.complex(vertical_ffted_real, vertical_ffted_imag) # (batch,c/2, h/2+1, w)
        vertical_output = torch.fft.irfft(vertical_ffted, n=(height), dim=(-2), norm=self.fft_norm)# (batch, c/2, h, w)

        return torch.cat([horizontal_output,vertical_output],dim=1)



class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.stride = stride
        self.conv1 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))
        self.fu = DualFourierUnit(
            out_channels // 2, out_channels // 2)
        self.conv2 = weight_norm(torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=1, bias=False))

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output



class ResidualBlock(nn.Module):
    def __init__(self, feats, distillation_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.distilled_channels = int(feats//2 * distillation_rate)
        self.remaining_channels = int(feats//2 - self.distilled_channels)
        self.c1 = weight_norm(nn.Conv2d(feats, feats//2, 3, 1, padding=1, groups=1, bias=False))
        self.c2 = weight_norm(nn.Conv2d(self.remaining_channels, feats//2, 3, 1, padding=1, groups=1, bias=False))
        self.c3 = weight_norm(nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, padding=1, groups=1, bias=False))
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = weight_norm(nn.Conv2d(self.distilled_channels * 3, feats, 3, 1, padding=1, groups=1, bias=False))

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        out = torch.cat([distilled_c1, distilled_c2, out_c3], dim=1)
        out_fused = self.c5(out) + input
        return out_fused



class GateAttn(nn.Module):
    def __init__(self, feats):
        super(GateAttn, self).__init__()
        self.spatial_attention = nn.Conv2d(feats*2, 2, 1, 1, padding=0, groups=1, bias=False)
        self.spatial_attention_hor = nn.Conv2d(2, 2, kernel_size=(1,5), stride=1, padding=(0,2), groups=1, bias=False)
        self.spatial_attention_vert = nn.Conv2d(2, 2, kernel_size=(5,1), stride=1, padding=(2,0), groups=1, bias=False)

    def forward(self, x):
        h,w = x.size(2),x.size(3)
        attn = self.spatial_attention(x)
        attn = self.spatial_attention_hor(attn) + self.spatial_attention_vert(attn)
        return attn



class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 dilation=1, bias=False, padding_type='reflect', gated=True):
        super(FFC, self).__init__()

        self.convs2s = ResidualBlock(feats = in_channels)
        self.convs2f = SpectralTransform(in_channels, out_channels, stride)
        self.convf2s = ResidualBlock(feats =in_channels)
        self.convf2f = SpectralTransform(in_channels, out_channels, stride)
        self.gated = gated
        if self.gated:
            self.gate = GateAttn(feats = in_channels)

    def forward(self, x):
        x_s, x_f = x
        if self.gated:
            total_input = torch.cat((x_s,x_f), dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            f2s_gate, s2f_gate = gates.chunk(2, dim=1)
        else:
            f2s_gate, s2f_gate  = 1, 1
        
        out_xs = self.convs2s(x_s) + self.convf2s(x_f) * f2s_gate
        out_xf = self.convs2f(x_s) * s2f_gate + self.convf2f(x_f)
        return out_xs, out_xf



class FFC_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=1, dilation=1, bias=False, activation_layer=nn.ReLU,
                 padding_type='reflect'):
        super(FFC_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, stride, padding, dilation,
                       bias, padding_type=padding_type)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self,  x ):
        x_s, x_f = x
        x_s, x_f = self.ffc((x_s, x_f))
        x_s = self.act(x_s)
        x_f = self.act(x_f)
        return x_s, x_f


class FFCResnetBlock(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = FFC_ACT(n_feat, n_feat, kernel_size=3, padding=1, dilation=1,
                                activation_layer=nn.LeakyReLU,
                                padding_type='reflect')
        self.conv2 = FFC_ACT(n_feat, n_feat, kernel_size=3, padding=1, dilation=1,
                                activation_layer=nn.LeakyReLU,
                                padding_type='reflect')
    def forward(self, x):
        x_s, x_f = x
        id_s, id_f = x_s, x_f
        x_s, x_f = self.conv1((x_s, x_f))
        x_s, x_f = self.conv2((x_s, x_f))
        x_s, x_f = id_s + x_s, id_f + x_f
        return x_s, x_f
