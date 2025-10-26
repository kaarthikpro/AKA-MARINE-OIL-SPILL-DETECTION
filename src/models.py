# src/models.py
import torch
import torch.nn as nn
from torchvision import models

# --- Utility for UNet ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- Segmentation Model: U-Net ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Simplified U-Net structure (as used in previous exchanges)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        b = self.bottleneck(self.pool(d3))
        
        u1 = self.up1(b)
        u1 = torch.cat((u1, d3), dim=1) # Skip connection
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d2), dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d1), dim=1)
        u3 = self.conv3(u3)

        return self.final(u3)

# --- Classification Model: ResNet50 ---
def get_classifier(num_classes=1, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    # Binary classification (Oil vs Non-Oil)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model