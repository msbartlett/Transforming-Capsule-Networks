
"""
PyTorch implementations of the residual block from "Deep Residual Learning for Image Recognition"
and MobileNetV2's inverted residual block from "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
"""
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)     # identity mapping
        out = F.relu(out)
        return out

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=1):
        super(InvertedResidualBlock, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            # Intermediate expansion layer
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, 1,
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            # Pointwise convolution
            nn.Conv2d(in_channels * expansion_factor, out_channels, 1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block