""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import pdb

from .unet_parts import *
import torch.nn as nn
from ..pinball import PinballLoss

class QWNet(nn.Module):
    def __init__(self, n_channels, n_classes, q_lo, q_hi, params, bilinear=True):
        super(QWNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.params = params
        self.bilinear = bilinear
        self.lhat = None
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

        # Three separate linear heads: one for each quantile, and one for the mean
        self.outc1 = OutConv(64, n_classes) # usually make this 64 in channels
        self.outc2 = OutConv(64, n_classes) # usually make this 64 in channels
        self.outc3 = OutConv(64, n_classes) # usually make this 64 in channels

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

        output = torch.cat((self.outc1(x), self.outc2(x), self.outc3(x)), dim=1)
        return output

    def loss_fn(self, pred, target):

        q_lo_loss = PinballLoss(quantile=self.q_lo)
        q_hi_loss = PinballLoss(quantile=self.q_hi)
        mse_loss = nn.MSELoss()

        loss = self.params['q_lo_weight'] * q_lo_loss(pred[:,0,:,:], target.squeeze()) + \
               self.params['q_hi_weight'] * q_hi_loss(pred[:,2,:,:], target.squeeze()) + \
               self.params['mse_weight'] * mse_loss(pred[:,1,:,:], target.squeeze())

        return loss

    def nested_sets(self, x, lam=None):
        if lam == None:
            if self.lhat == None:
                raise Exception("You have to specify lambda unless your model is already calibrated.")
            lam = self.lhat 
        output = self(x)
        return self.nested_sets_from_output(output,lam=lam)

    def nested_sets_from_output(self, output, lam=None):
        if lam == None:
            if self.lhat == None:
                raise Exception("You have to specify lambda unless your model is already calibrated.")
            lam = self.lhat 
        upper_edge = lam * (output[:,2,:,:] - output[:,1,:,:]) + output[:,1,:,:] 
        lower_edge = output[:,1,:,:] - lam * (output[:,1,:,:] - output[:,0,:,:])
        upper_edge = torch.maximum(upper_edge, output[:,1,:,:] + 1e-1) # set a lower bound on the size.
        lower_edge = torch.minimum(lower_edge, output[:,1,:,:] - 1e-1)
        return lower_edge, upper_edge 
