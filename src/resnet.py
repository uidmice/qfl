import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, bias=False)

        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=1, bias=False)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # When dimensions do not match, we apply a 1x1 convolution.
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(ResLayer, self).__init__()
        layers = []
        # First block may downsample if stride != 1.
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks keep the same dimensions.
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)


class ResNet18Tiny(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18Tiny, self).__init__()
        # Modified initial conv for 64x64 input.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        # No initial maxpool
        
        # Using the ResLayer class to build residual layers:
        self.layer1 = ResLayer(64, 64, num_blocks=2, stride=1)   # 64x64
        self.layer2 = ResLayer(64, 128, num_blocks=2, stride=2)  # 32x32
        self.layer3 = ResLayer(128, 256, num_blocks=2, stride=2) # 16x16
        self.layer4 = ResLayer(256, 512, num_blocks=2, stride=2) # 8x8
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)     # (B, 64, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool
        
        x = self.layer1(x)    # (B, 64, 64, 64)
        x = self.layer2(x)    # (B, 128, 32, 32)
        x = self.layer3(x)    # (B, 256, 16, 16)
        x = self.layer4(x)    # (B, 512, 8, 8)
        
        x = self.avgpool(x)   # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.fc(x)        # (B, num_classes)
        return x