import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.network import (
    QuadraticConnectionUnitS,
    AdditionFusionS,
    ResBlockS
)


class PrePyramidL1S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL1S, self).__init__()
        self.conv_first = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        feat_l1 = self.resblock(feat_l1)
        return feat_l1



class PrePyramidL2S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL2S, self).__init__()
        self.conv_first = nn.Conv2d(1, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        feat_l2 = self.conv_first(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        _, _, h, w = x.size()
        feat_l2 = nn.Upsample((h, w), mode='bilinear', align_corners=False)(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        return feat_l2


class SYESRX8NetS(nn.Module):
    def __init__(self, channels):
        super(SYESRX8NetS, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.headpre = AdditionFusionS(PrePyramidL1S(1), PrePyramidL2S(1), 1)
        self.resblock = ResBlockS(num_feat=1)
        self.head = QuadraticConnectionUnitS(
            nn.Sequential(
                nn.Conv2d(1, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            nn.Conv2d(1, channels, 5, 1, 2),
            channels
        )
        self.body = QuadraticConnectionUnitS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, ),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels,  channels, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.tail = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 1, 3, 1, 1)
        )

    def forward(self, x):
        inp = x
        x = self.headpre(x)
        x = self.resblock(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=8, mode='bilinear', align_corners=False)
        x = self.tail(x) + base
        return x 
