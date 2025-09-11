import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, padding_mode='zeros'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
    
class ResNet2D(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels, padding_mode='zeros'):
        """
        Args:
            in_channels (int): Number of input channels.
            base_channels (List[int]): Number of channels for each residual block layer.
            out_channels (int): Number of output channels.
            padding_mode (str): Padding mode ('zeros', 'reflect', etc.)
        """
        super().__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels[0], kernel_size=3, padding=1, padding_mode=padding_mode)
        self.initial_bn = nn.BatchNorm2d(base_channels[0])

        # Residual blocks
        blocks = []
        for i in range(len(base_channels) - 1):
            blocks.append(BasicResidualBlock(base_channels[i], base_channels[i + 1], padding_mode=padding_mode))
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Conv2d(base_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x