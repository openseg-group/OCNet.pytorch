import torch
import torch.nn as nn
import torch.nn.functional as F
affine_par = True

class Separable_transpose_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, output_padding=0, bias=False, dilation=1):
        super(Separable_transpose_convolution, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding, output_padding, groups=in_channels, bias=bias, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=affine_par)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size):
        x = self.relu(self.bn1(self.conv1(x, output_size)))
        x = self.bn2(self.conv2(x))
        return x


class Separable_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, dilation=1):
        super(Separable_convolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=affine_par)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x