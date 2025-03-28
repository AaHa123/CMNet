from functools import partial
import torch.nn.functional as F
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
from net.PVTv2 import pvt_v2_b4
from net.PVTv2 import pvt_v2_b2
import numpy as np
import torch
from torch import nn

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

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        #hf = F.interpolate(hf, size, mode='bilinear', align_corners=False)
        x4 = self.reduce4(x4)
        x1 = self.reduce1(x1)
        o1 = self.block(torch.cat([x4, x1], 1))
        return o1

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
'''
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
'''
class DFEM(nn.Module):
    def __init__(self,channel,out_channel,need_relu=True):
        super(DFEM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(channel, out_channel, kernel_size=1, stride=1)
        )
        self.conv1_1 = Conv1x1(channel,out_channel)
        inter_channel = out_channel//2

        self.conv0a = BasicConv2d(inter_channel,inter_channel,1)
        self.conv0b = ConvBNR(inter_channel,inter_channel,3)
        self.convb1 = ConvBNR(inter_channel,inter_channel,3)
        self.convb2 = ConvBNR(inter_channel,inter_channel,3)
        self.convb3 = ConvBNR(inter_channel,inter_channel,3)
        self.conv0_0 = BasicConv2d(5*inter_channel, out_channel,1)
        self.relu = nn.ReLU(True)
        self.conv_res = BasicConv2d(channel, out_channel, 1)
        self.conv0 = BasicConv2d(channel, out_channel, 3, padding=1)
        self.conv1 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)


    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x) #池化 卷积--全局
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True) #统一尺寸
        xc = self.conv1_1(x) # reduce 256--64
        #print('xc',xc.shape)
        xc = torch.chunk(xc, 2, dim=1) # 沿着通道分割4 64/4=14
        # 通道内依赖关系 --探索细节语义
        xa = self.conv0a(xc[0]) #//2
        #print('xa',xa.shape)
        xb = self.conv0b(xc[1]) #//2
        xb1 = self.convb1(xb) 
        xb2 = self.convb2(xb1)
        xb3 = self.convb3(xb2)
        xx = self.conv0_0(torch.cat((xa, xb, xb1, xb2,xb3), dim=1))
        #print('xx',xx.shape)
        
        # 探索整体的上下文
        x20 = self.conv0(x)
        #print('20',x20.shape)
        x21 = x20 + self.conv1(x20)
        x22 = x21 + self.conv2(x21)
        x23 = x22 + self.conv3(x22)
        #print('23',x23.shape)
        x_cat = self.conv_cat(torch.cat((x20, x21, x22, x23), 1)) 
        #print('x_cat',x_cat.shape)

        x = self.relu(x_cat + xx + self.conv_res(x))
        return x
        
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
        out1 = x1 * weight1 
        out2 = x2 * weight2
        out3 = x3 * weight3
        return out1, out2, out3

class SCAM(nn.Module):
    def __init__(self, channel):
        super(SCAM, self).__init__()
        self.conv1 = Conv1x1(channel*2, channel)
        #ConvBNR(channel*2,channel,3)
        #self.conv2 = FOM(3 * channel, channel, False)
        self.sea = SEAttention(channel)
        self.conv3 = BasicConv2d(channel*3, channel, 3, padding=1)

    def forward(self, x, y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        k = self.conv1(torch.cat((x, y), 1))
        z = x + y * k
        #b = self.conv2(torch.cat((x, y, z), 1))
        x,y,z = self.sea(x,y,z) #通道 计算交换性
        ret = self.conv3(torch.cat([x,y,z],1))  # 交互特征拼接，突出细节
        return ret
    
class MFM(nn.Module):
    def __init__(self, channel):
        super(MFM, self).__init__()
        self.conv1_1 = Conv1x1(channel*2, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(x0 + xc[2])
        x2 = self.dconv7_1(x1 + xc[3])
        x3 = self.dconv9_1(x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
    
class UnNamedModule(nn.Module):
    def __init__(self, channel):
        super(UnNamedModule, self).__init__()
        self.fia1 = MFM(channel)
        self.fia2 = MFM(channel)
        self.fia3 = MFM(channel)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 1)
        self.conv2 = nn.Conv2d(channel, 1, 1)
    def forward(self, x1, x2, x3, x4):
        x2a = self.fia1(x1, x2) #upsample上采样，统一hw
        x3a = self.fia2(x2a, x3)
        x4a = self.fia3(x3a,x4)
        x = self.conv1(x4a) #1*1 卷积
        ret = self.conv2(x)
        return ret
    
class CGM(nn.Module):
    def __init__(self, channel):
        super(CGM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k,
                                padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fom = FOM(1, 1)
        self.sigmoid_coarse = nn.Sigmoid()

    def forward(self, c, coarse_att):
        if c.size() != coarse_att.size():
            coarse_att = F.interpolate(coarse_att, c.size()[2:], mode='bilinear', align_corners=False) #下采样
        
        coarse_att2 = self.sigmoid_coarse(coarse_att)
        x = c * coarse_att2 + c
        y = c - (1- coarse_att2)*c
        # ECA 通道注意力机制
        x = self.conv2d(x+y)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        return c + x
    
class Net(nn.Module):
    def __init__(self,channel=64):
        super(Net, self).__init__()
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.context_encoder = pvt_v2_b2(pretrained=True)

        self.d1 = DFEM(64, channel)
        self.d2 = DFEM(128, channel)
        self.d3 = DFEM(320, channel)
        self.d4 = DFEM(512, channel)

        #self.edge = EAM()
        self.um = UnNamedModule(channel)

        self.cam1 = SCAM(channel)
        self.cam2 = SCAM(channel)
        self.cam3 = SCAM(channel)

        self.predictor1 = nn.Conv2d(channel, 1, 1)
        self.predictor2 = nn.Conv2d(channel, 1, 1)
        self.predictor3 = nn.Conv2d(channel, 1, 1)
        self.igm1 = CGM(channel)
        self.igm2 = CGM(channel)
        self.igm3 = CGM(channel)
        self.igm4 = CGM(channel)

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

        # DFEM
        d1 = self.d1(x1)
        d2 = self.d2(x2)
        d3 = self.d3(x3)
        d4 = self.d4(x4)

        #HPM+GRM = GRM
        coarse_att = self.um(d1, d2, d3, d4)
        g1 = self.igm1(d1,coarse_att)
        g2 = self.igm1(d2,coarse_att)
        g3 = self.igm1(d3,coarse_att)
        g4 = self.igm1(d4,coarse_att)

        # MFM
        x34 = self.cam3(g3, g4)
        x234 = self.cam2(g2, x34)
        x1234 = self.cam1(g1, x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, input_size, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, input_size, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, input_size, mode='bilinear', align_corners=False)

        # o11 = self.predictor1(pg1)
        oc = F.interpolate(coarse_att, scale_factor=4,
                           mode='bilinear', align_corners=False)

        return o3, o2, o1, oc
