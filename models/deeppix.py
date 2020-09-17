import torch
import torch.nn.functional as F
import torch.nn as nn
from losses.loss import AngularLoss, AngularConvLoss
from torchvision import models


class DeepPixBiS(nn.Module):

    def __init__(self, pretrained=True):
        super(DeepPixBiS, self).__init__()
        self.pretrained = pretrained
        self.base_line = BaseLineConv(self.pretrained)
        self.final_op = AngularLoss(512, 1, loss_type='binary-angularloss')
        self.ang_conv = AngularConvLoss(384, 1, loss_type='binary-conv-angularloss')

    def forward(self, x, labels_binary=None, labels_map=None, embed=False):
        projection, enc = self.base_line(x)
        if embed:
            return projection
        L1, sig = self.final_op(projection, labels_binary)
        L2, feat_map = self.ang_conv(enc, labels_map)
        return L1, L2, feat_map, sig


class BaseLineConv(nn.Module):

    def __init__(self, pretrained=False):
        super(BaseLineConv, self).__init__()
        base_line = models.densenet161(pretrained=pretrained)
        feature_extractor = list(base_line.children())
        self.backbone = nn.Sequential(*feature_extractor[0][0:7])
        self.enc = nn.Sequential(*feature_extractor[0][7])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, padding=0)
        self.linear = nn.Linear(14 * 14, 512)

    def forward(self, x):
        # pass the 224x224 image through the base model
        x = self.backbone(x)
        
        enc = self.enc(x)
        
        # generate a 14x14 map
        dec = self.dec(enc)
        
        # final embedding layer
        dec_flat = dec.view(-1, 14 * 14)
        projection = self.linear(dec_flat)
        
        return projection, enc
