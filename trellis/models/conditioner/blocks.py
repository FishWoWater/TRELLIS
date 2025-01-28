import torch
import torch.nn as nn


class VolumeProcessorBlock(nn.Module):

    def __init__(self, x_in_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(16, c)
        self.bn = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_in_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(self.silu(self.bn(x)))


class VolumeFeatureExtractionNetwork(nn.Module):

    def __init__(self, input_dim=1024, dims=(512, 1024)):
        super().__init__()
        d0, d1 = dims

        self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
        self.conv0 = VolumeProcessorBlock(d0, d0, stride=1)

        self.conv1 = VolumeProcessorBlock(d0, d1, stride=2)
        self.conv2_0 = VolumeProcessorBlock(d1, d1, stride=1)
        self.conv2_1 = VolumeProcessorBlock(d1, d1, stride=1)

    def forward(self, x):
        x = self.init_conv(x)
        conv0 = self.conv0(x)

        x = self.conv1(conv0)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        return x
