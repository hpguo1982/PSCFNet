import pvtv2
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pvtv2 import pvt_v2_b3
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CFN(nn.Module):
    def __init__(self, channel):
        super(CFN, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class multikernel_dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.c3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.c4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.s1 = Conv2D(out_c * 4, out_c, kernel_size=1, padding=0)

        self.d1 = Conv2D(out_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.d2 = Conv2D(out_c, out_c, kernel_size=3, padding=3, dilation=3)
        self.d3 = Conv2D(out_c, out_c, kernel_size=3, padding=7, dilation=7)
        self.d4 = Conv2D(out_c, out_c, kernel_size=3, padding=11, dilation=11)
        self.s2 = Conv2D(out_c * 4, out_c, kernel_size=1, padding=0, act=False)
        self.s3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s1(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s2(x)
        s = self.c3(x0)

        x = self.relu(x + s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x


class CSAB(nn.Module):
    def __init__(self, in_channel):
        super(CSAB, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        x1_ca = x.mul(self.ca(x))
        x1_sa = x1_ca.mul(self.sa(x1_ca))
        x = x + x1_sa
        return x

class SAB(nn.Module):
    def __init__(self, channels, padding=0, groups=1, matmul_norm=True):
        super(SAB, self).__init__()
        self.channels = channels
        self.padding = padding
        self.groups = groups
        self.matmul_norm = matmul_norm
        self._channels = channels//8

        self.conv_query = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_key = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups)

        self.conv_output = Conv2D(in_c=channels, out_c=channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Get query, key, value tensors
        query = self.conv_query(x).view(batch_size, -1, height*width)
        key = self.conv_key(x).view(batch_size, -1, height*width)
        value = self.conv_value(x).view(batch_size, -1, height*width)

        # Apply transpose to swap dimensions for matrix multiplication
        query = query.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels//8)
        value = value.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels)

        # Compute attention map
        attention_map = torch.matmul(query, key)
        if self.matmul_norm:
            attention_map = (self._channels**-.5) * attention_map
        attention_map = torch.softmax(attention_map, dim=-1)

        # Apply attention
        out = torch.matmul(attention_map, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)

        # Apply output convolution
        out = self.conv_output(out)
        out = out + x

        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PSCFNet(nn.Module):
    def __init__(self, channel=32):
        super(PSCFNet, self).__init__()

        self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
        path = 'pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.c11 = multikernel_dilated_conv(64, 64)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()


        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.r1 = ResidualBlock(64, 64)
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        cim_feature = self.c11(x1)

        # CFN
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        cfm_feature = self.CFN(x4_t, x3_t, x2_t,)


        T21 = self.Translayer2_0(cim_feature)
        T22 = self.down05(T21)
        x = torch.cat([cfm_feature, T22], axis=1)
        x11 = self.r1(x)
        x22 = self.up(x11)
        y = self.y(x22)

        return y


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = PSCFNet()
    y = model(x)
    print(y.shape)
