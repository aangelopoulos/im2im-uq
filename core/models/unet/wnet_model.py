""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import pdb

from .unet_parts import *


class WNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(WNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # path 1
        self.p1inc = DoubleConv(n_channels, 32)
        self.p1down1 = Down(32, 64)
        self.p1down2 = Down(64, 128)
        self.p1down3 = Down(128, 256)
        self.p1down4 = Down(256, 512 // factor)
        # path 2
        self.p2inc = DoubleConv(n_channels, 32)
        self.p2down1 = Down(32, 64)
        self.p2down2 = Down(64, 128)
        self.p2down3 = Down(128, 256)
        self.p2down4 = Down(256, 512 // factor)

        # joined path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.histology = Up_custom(64, 32, bilinear, (8.0 / 3.0))
        self.outc = OutConv(64, n_classes) # usually make this 64 in channels

    def forward(self, x):
        p1, p2 = (x[:,0:1,:,:], x[:,1:2,:,:])
        p1_1 = self.p1inc(p1)
        p1_2 = self.p1down1(p1_1)
        p1_3 = self.p1down2(p1_2)
        p1_4 = self.p1down3(p1_3)
        p1_5 = self.p1down4(p1_4)

        p2_1 = self.p2inc(p2)
        p2_2 = self.p2down1(p2_1)
        p2_3 = self.p2down2(p2_2)
        p2_4 = self.p2down3(p2_3)
        p2_5 = self.p2down4(p2_4)
        x = torch.cat((p1_5, p2_5), dim=1)

        x = self.up1(x, torch.cat((p1_4, p2_4), dim=1))
        x = self.up2(x, torch.cat((p1_3, p2_3), dim=1))
        x = self.up3(x, torch.cat((p1_2, p2_2), dim=1))
        x = self.up4(x, torch.cat((p1_1, p2_1), dim=1))
        #x = self.histology(x)
        logits = self.outc(x)
        return logits
