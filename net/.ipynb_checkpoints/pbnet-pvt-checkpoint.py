from functools import partial

import torch.nn.functional as F
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
from net.PVTv2 import pvt_v2_b4
from net.PVTv2 import pvt_v2_b2
import numpy as np
import torch
from torch import nn
from .util import wavelet


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0,
                 need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding,
                                       bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))
        self.block2 = nn.Sequential(
            BasicConv2d(96+256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x4 = self.reduce4(x4)
        #x1 = self.reduce1(x1)
        o1 = self.block(torch.cat([x4, x1], 1))
        
        return o1

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False),
                nn.LeakyReLU(0.1, inplace=True)
            )
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        edge_info = []

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop() #3个
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            edge_info.append(curr_x_h)

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x,edge_info[0]
    
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class FOM(nn.Module):
    def __init__(self, in_channel, out_channel, need_relu=True):
        super(FOM, self).__init__()
        self.need_relu = need_relu
        self.relu = nn.ReLU(True)
        self.conv0 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv1 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = x0 + self.conv1(x0)
        x2 = x1 + self.conv2(x1)
        x3 = x2 + self.conv3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        if self.need_relu:
            x = self.relu(x_cat + self.conv_res(x))
        else:
            x = x_cat + self.conv_res(x)
        return x

class FEHM(nn.Module):
    def __init__(self,channel,out_channel,need_relu=True):
        super(FEHM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(channel, out_channel, kernel_size=1, stride=1)
        )
        self.conv1_1 = Conv1x1(channel,out_channel)
        inter_channel = out_channel//4
        self.conv3_1 = ConvBNR(inter_channel, inter_channel, 3)
        self.dconv5_1 = ConvBNR(inter_channel, inter_channel, 3, dilation=2)
        self.dconv7_1 = ConvBNR(inter_channel, inter_channel, 3, dilation=3)
        self.dconv9_1 = ConvBNR(inter_channel, inter_channel, 3, dilation=4)
        self.conv1_2 = Conv1x1(out_channel, out_channel)
        self.relu = nn.ReLU(True)
        self.conv_res = BasicConv2d(channel, out_channel, 1)
        self.conv0 = BasicConv2d(channel, out_channel, 3, padding=1)
        self.conv1 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)


    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x) #池化 卷积--全局
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True) #统一尺寸
        xc = self.conv1_1(x) # reduce 256--64
        xc = torch.chunk(xc, 4, dim=1) # 沿着通道分割4 64/4=14
        # 通道内依赖关系 --探索细节语义
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        # 探索整体的上下文
        x20 = self.conv0(x)
        x21 = x20 + self.conv1(x20)
        x22 = x21 + self.conv2(x21)
        x23 = x22 + self.conv3(x22)
        x_cat = self.conv_cat(torch.cat((x20, x21, x22, x23, xx), 1))

        x = self.relu(x_cat + self.conv_res(x))

        return x
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        # 通过卷积操作生成空间注意力图
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: 输入特征图, 形状为 (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均池化和最大池化结果拼接在一起
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积操作生成空间注意力图
        attention_map = self.sigmoid(self.conv1(x_cat))
        # 逐元素乘法应用空间注意力
        out = x * attention_map
        return out
    

class FTM(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(FTM, self).__init__()
        self.sea = SEAttention(out_channels)
        # self.conv = ConvBNR(out_channels*2, out_channels,3)
        self.conv = BasicConv(2 * in_channel, out_channels, 3, 1, padding=(3 - 1) // 2, relu=False)
        #self.conv1 = Conv1x1(out_channels * 3, out_channels)
        self.att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)
        )
        self.sp = SpatialAttention()

    def forward(self, e1, e2, e3, e4):
        lh1, hl1, hh1 = e1[:, :, 0, :, :], e1[:, :, 1, :, :], e1[:, :, 2, :, :]  # torch.Size([16, 256, 52, 52])

        # 将高频分量拼接在一起并进行上采样
        hh_size = hh1.size()[2:]
        lh, hl, hh = [], [], []
        es = [e1, e2, e3, e4]
        for i, e in enumerate(es):
            lh.append(
                F.interpolate(e[:, :, 0, :, :], size=hh_size, mode='bilinear', align_corners=False))
            hl.append(
                F.interpolate(e[:, :, 1, :, :], size=hh_size, mode='bilinear', align_corners=False))
            hh.append(
                F.interpolate(e[:, :, 2, :, :], size=hh_size, mode='bilinear', align_corners=False))

            # 融合高频信息
        lh_stack = sum(torch.stack(lh))
        hl_stack = sum(torch.stack(hl))
        hh_stack = sum(torch.stack(hh))

        hf_fused = self.sea(lh_stack, hl_stack, hh_stack)

        return hf_fused

class HFG(nn.Module):
    def __init__(self, channel):
        super(HFG, self).__init__()
        self.conv_final = nn.Sequential(
            Conv1x1((channel + 64), (channel + 64) // 2),
            nn.Conv2d((channel + 64) // 2, channel, 1))

    def forward(self, t1, HH):
        # print(t1.shape)
        hh_up = F.interpolate(HH, size=t1.size()[2:], mode='bilinear')  # t1 torch.Size([16, 64,)
        t1_hh = self.conv_final(torch.cat([hh_up, t1], 1))  # tt1_hh torch.Size([16, 256])
        return t1_hh

from torch.nn import init
class SEAttention(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 3*channel, bias=False),
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1,x2,x3):
        # (B,C,H,W)
        B, C, H, W = x1.size()
        x = x1 + x2 + x3
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,3C)-->(B, 3C, 1, 1)
        y = self.fc(y).view(B, 3*C, 1, 1)
        # split
        weight1 = torch.sigmoid(y[:,:C,:,:]) # (B,C,1,1)
        weight2 = torch.sigmoid(y[:, C:2*C, :, :]) # (B,C,1,1)
        weight3 = torch.sigmoid(y[:, 2*C:, :, :]) # (B,C,1,1)
        # scale: (B,C,H,W) * (B,C,1,1) == (B,C,H,W)
        out = x1 * weight1 + x2 * weight2 + x3 * weight3
        return out
#    Position Attention Module (PAM)

class FullImageAttention(nn.Module):
    def __init__(self, channel):
        super(FullImageAttention, self).__init__()
        self.conv1 = FOM(channel*2, channel)
        #self.conv2 = FOM(3 * channel, channel, False)
        self.sea = SEAttention(channel)
        self.conv3 = BasicConv2d(channel*2, channel, 3, padding=1)

    def forward(self, x, y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        k = self.conv1(torch.cat((x, y), 1))
        z = x + y * k
        #b = self.conv2(torch.cat((x, y, z), 1))
        b = self.sea(x,y,z) #通道 计算交换性
        ret = self.conv3(torch.cat([b,z],1))  # 交互特征拼接，突出细节
        return ret
    
class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        # self.getalpha = getAlpha(in_channels, in_channels,kernel_size=1, bias=False)
        self.getalpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),  # 使用 out_channels
            nn.Sigmoid()
        )

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1), )
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        return out
    
class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out

class APM(nn.Module):
    def __init__(self, in_channels, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        super(APM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv1x1(in_channels, depth)
        )
        self.branch0 = BasicConv2d(in_channels, depth, kernel_size=1, stride=1)
        inter_channnel = depth//4
        self.branch1 = BasicConv2d(inter_channnel, inter_channnel, 3, padding=1)
        self.branch2 = BasicConv2d(inter_channnel, inter_channnel, 3, padding=1)
        self.branch3 = BasicConv2d(inter_channnel, inter_channnel, 3, padding=1)
        self.branch4 = BasicConv2d(inter_channnel, inter_channnel, 3, padding=1)
        self.head = nn.Sequential(
            BasicConv2d(depth * 2, 128, kernel_size=3, padding=1),
            PAM(128)
        )
        self.out = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)  # 池化 卷积--全局信息
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)  # 统一尺寸
        branch0 = self.branch0(x)  # 1*1 卷积 降低通道
        branch = torch.chunk(branch0, 4, 1)
        branch1 = self.branch1(branch[0])
        branch2 = branch1 + self.branch2(branch[1])
        branch3 = branch2 + self.branch3(branch[2])
        branch4 = branch3 + self.branch4(branch[3])
        out = torch.cat([branch1, branch2, branch3,branch4], 1)+branch0 # 类似残差思想
        out = torch.cat([branch_main,out],1) #全局，捕捉长距离依赖
        out = self.head(out)
        out = self.out(out)
        return out
    
class PFM(nn.Module):
    def __init__(self, channel):
        super(PFM, self).__init__()
        self.conv = nn.Conv2d(channel * 2, channel, kernel_size=1)
        self.ode = ODE(channel)
        self.out_y = nn.Sequential(
            ConvBNR(channel * 2, channel, kernel_size=3),
            ConvBNR(channel, channel // 2, kernel_size=3),
            nn.Conv2d(channel // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, pos):
        prior_cam = F.interpolate(pos, size=x.size()[2:], mode='bilinear',align_corners=True)  # 2,1,12,12->2,1,48,48
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1)) #初步
        ode_out = self.ode(yt) #细化
        cat2 = torch.cat([yt, ode_out], dim=1)  # 2,128,48,48
        y = self.out_y(cat2)
        y = y + prior_cam
        return y
    
class MIGM(nn.Module):
    def __init__(self, channel):
        super(MIGM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fom_y = FOM(channel,channel)


    def forward(self, c, edge_att,pos):
        ## edge_guided
        if c.size() != edge_att.size() or c.size()!= pos.size():
            edge_att = F.interpolate(edge_att, c.size()[2:], mode='bilinear', align_corners=False)
            pos = F.interpolate(pos, c.size()[2:], mode='bilinear', align_corners=True)
        pos = self.sigmoid(pos)
        # 前景
        y = c * edge_att + pos * c +c# 突出前景？
        y = self.fom_y(y)
        g = c - (1 - pos) * c
        x = self.conv2d(g + y)
        wei = self.avg_pool(g)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        out = x * wei + c
        return out
    


class Net(nn.Module):
    def __init__(self,channel=64):
        super(Net, self).__init__()
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.context_encoder = pvt_v2_b4(pretrained=True)
        # if self.training:
        # self.initialize_weights()
        self.et1 = FEHM(64, channel)
        self.et2 = FEHM(128, channel)
        self.et3 = FEHM(320, channel)
        self.et4 = FEHM(512, channel)

        self.wt1 = WTConv2d(in_channels=channel, out_channels=channel)
        self.wt2 = WTConv2d(in_channels=channel, out_channels=channel)
        self.wt3 = WTConv2d(in_channels=channel, out_channels=channel)
        self.wt4 = WTConv2d(in_channels=channel, out_channels=channel)

        self.edge = EAM()
        self.reduce1 = Conv1x1(256, 64)
        self.ft = FTM(channel, channel)
        self.hfg = HFG(channel)

        self.cam1 = FullImageAttention(channel)
        self.cam2 = FullImageAttention(channel)
        self.cam3 = FullImageAttention(channel)

        self.apm = APM(channel)
        self.pfm1 = PFM(channel)
        self.pfm2 = PFM(channel)
        self.pfm3 = PFM(channel)
        self.pfm4 = PFM(channel)

        self.migm1 = MIGM(channel) 
        self.migm2 = MIGM(channel)
        self.migm3 = MIGM(channel)
        self.migm4 = MIGM(channel)

        self.predictor1 = nn.Conv2d(channel, 1, 1)
        self.predictor2 = nn.Conv2d(channel, 1, 1)
        self.predictor3 = nn.Conv2d(channel, 1, 1)

    # def initialize_weights(self):
    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        input_size = x.size()[2:]
        #x1, x2, x3, x4 = self.resnet(x)
        endpoints = self.context_encoder.extract_endpoints(x)
        x1 = endpoints['reduction_2']  # torch.Size([16, 64, 104, 104])
        x2 = endpoints['reduction_3']  # [16, 128, 52, 52])
        x3 = endpoints['reduction_4']  # [16, 320, 26, 26])
        x4 = endpoints['reduction_5']  # [16, 512, 13, 13])

        t1 = self.et1(x1)
        t2 = self.et2(x2)
        t3 = self.et3(x3)
        t4 = self.et4(x4)

        _, e1 = self.wt1(t1)
        _, e2 = self.wt2(t2)
        _, e3 = self.wt3(t3)
        f4, e4 = self.wt4(t4)  

        HH = self.ft(e1, e2, e3, e4)
        th1 = self.hfg(t1, HH)  
        edge = self.edge(x4, x1)  
        edge_att = torch.sigmoid(edge)

        pos = self.apm(f4) # 给大感受野 多尺度交互性的特征生成，全局上 锁定位置
        pg3 = self.pfm3(t3, pos)
        pg2 = self.pfm2(t2, pg3)
        pg1 = self.pfm1(th1, pg2)  # 保留更多边缘纹理

        x1a = self.migm1(t1, edge_att, pg1)
        x2a = self.migm2(t2, edge_att, pg1)
        x3a = self.migm3(t3, edge_att, pg1)
        x4a = self.migm4(t4, edge_att, pg1)

        x34 = self.cam3(x3a, x4a)
        x234 = self.cam2(x2a, x34)
        x1234 = self.cam1(x1a, x234)
        
        o33 = F.interpolate(pg3, input_size, mode='bilinear', align_corners=True)
        o22 = F.interpolate(pg2, input_size, mode='bilinear' , align_corners=True)
        o11 = F.interpolate(pg1, input_size, mode='bilinear', align_corners=True)
        #o44 = F.interpolate(pg4, input_size, mode='bilinear', align_corners=True)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, input_size, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, input_size, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, input_size, mode='bilinear', align_corners=False)

        # o11 = self.predictor1(pg1)
        oe = F.interpolate(edge_att, input_size, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe, [o33, o22, o11]


