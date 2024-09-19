# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 01:45:32 2024

@author: jishu
"""
import torch
import torch.nn as nn

import torchvision.models as models

class UNet(nn.Module):
    
    def __init__(self, n_classes=2):
        super(UNet, self).__init__()
        
        self.n_classes = n_classes
        self.pretrained_backbone = models.resnet18(pretrained=True)
        self.layers = list(self.pretrained_backbone.children())
        
       
        self.downsample_1 = nn.Sequential(*self.layers[0:3])  # 1 x 64 x 112 x 112
        self.downsample_2 = nn.Sequential(*self.layers[3:5])  # 1 x 64 x 56 x 56
        self.downsample_3 = self.layers[5]  # 1 x 128 x 28 x 28
        self.downsample_4 = self.layers[6]  # 1 x 256 x 14 x 14
        self.downsample_5 = self.layers[7]  # 1 x 512 x 7 x 7
        
       
        self.up_conv_1 = self.upConv(512, 256)
        self.decoder_1 = self.DoubleConv(512, 256)
        
        self.up_conv_2 = self.upConv(256, 128)
        self.decoder_2 = self.DoubleConv(256, 128)
        
        self.up_conv_3 = self.upConv(128, 64)
        self.decoder_3 = self.DoubleConv(128, 64)  # 1 x 64 x 56 x 56
        
        self.up_conv_4 = self.upConv(64, 64)
        self.decoder_4 = self.DoubleConv(128, 64)  # 1 x 64 x 112 x 112
        
        self.up_conv_5 = self.upConv(64, 64)  # 1 x 64 x 224 x 224
        
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=1)
        
    def forward(self, x):
        
        enc1 = self.downsample_1(x)  # 1 x 64 x 112 x 112
        enc2 = self.downsample_2(enc1)  # 1 x 64 x 56 x 56
        enc3 = self.downsample_3(enc2)  # 1 x 128 x 28 x 28
        enc4 = self.downsample_4(enc3)  # 1 x 256 x 14 x 14
        enc5 = self.downsample_5(enc4)  # 1 x 512 x 7 x 7
        
        up1 = self.up_conv_1(enc5)  # 1 x 256 x 14 x 14
        dec1 = self.decoder_1(torch.cat([enc4, up1], dim=1))  # 1 x 256 x 14 x 14
        
        up2 = self.up_conv_2(dec1)  # 1 x 128 x 28 x 28
        dec2 = self.decoder_2(torch.cat([enc3, up2], dim=1))  # 1 x 128 x 28 x 28
        
        up3 = self.up_conv_3(dec2)  # 1 x 64 x 56 x 56
        dec3 = self.decoder_3(torch.cat([enc2, up3], dim=1))  # 1 x 64 x 56 x 56
        
        up4 = self.up_conv_4(dec3)  # 1 x 64 x 112 x 112
        dec4 = self.decoder_4(torch.cat([enc1, up4], dim=1))  # 1 x 64 x 112 x 112
        
        up5 = self.up_conv_5(dec4)  # 1 x 64 x 224 x 224

        output = self.final(up5)

        
        return output

    def DoubleConv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)
        )

    def upConv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
