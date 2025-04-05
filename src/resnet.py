import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.qm import *

class BasicBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
      
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # When dimensions do not match, we apply a 1x1 convolution.
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False)
            
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)        
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    
class QBasicBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride, kernel_size, padding,
                  quantizer, weight_update, forward_rescale, backward_rescale):
        super().__init__()
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size,
                             stride, padding, quantizer, weight_update)
        self.bn1 = QBatchNorm2d(out_channels,  weight_update, forward_rescale, backward_rescale)
        self.relu = QReLU(forward_rescale, backward_rescale)

        self.conv2 = QConv2d(out_channels, out_channels, kernel_size,
                             1, 1, quantizer, weight_update)
        self.bn2 = QBatchNorm2d(out_channels,  weight_update, forward_rescale, backward_rescale)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # When dimensions do not match, we apply a 1x1 convolution.
            self.downsample =QConv2d(in_channels, out_channels, 1,
                             stride, 0, quantizer, weight_update)
        self.backward_rescale = backward_rescale

    def forward(self, x):
        x0, s0 = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)        
        x1, s1 = self.bn2(out)
        if self.downsample is not None:
            x0, s0 = self.downsample(x)
        
        out = x1 * s1 + x0 * s0
        out = out, 1
        return self.relu(out)
    
    def backward(self, err):
        e = self.relu.backward(err)

        e1 = self.bn2.backward(e)
        e1 = self.conv2.backward(e1)
        e1 = self.relu.backward(e1)
        e1 = self.bn1.backward(e1)
        e1 = self.conv1.backward(e1)
        if self.downsample is not None:
            e = self.downsample.backward(e)
        
        e = e1[0] * e1[1] + e[0] * e[1]
        return self.backward_rescale(e, 1)
    
    def to(self, device):
        super().to(device)
        self.conv1.to(device)
        self.bn1.to(device)
        self.relu.to(device)
        self.conv2.to(device)
        self.bn2.to(device)
        if self.downsample is not None:
            self.downsample.to(device)
            
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

    
class QResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, kernel_size, padding,
                  quantizer, weight_update, forward_rescale, backward_rescale ):
        super(QResLayer, self).__init__()
        layers = []
        layers.append(QBasicBlock(in_channels, out_channels, stride, kernel_size, padding,
                  quantizer, weight_update, forward_rescale, backward_rescale))
        # Remaining blocks keep the same dimensions.
        for _ in range(1, num_blocks):
            layers.append(QBasicBlock(out_channels, out_channels, 1, kernel_size, padding,
                  quantizer, weight_update, forward_rescale, backward_rescale))
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)
    
    def backward(self, err):
        for block in reversed(self.blocks):
            err = block.backward(err)
        return err
    
    def to(self, device):
        super().to(device)
        for block in self.blocks:
            block.to(device)


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