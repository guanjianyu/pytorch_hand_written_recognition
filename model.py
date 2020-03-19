"""
This File contained pytorch model for handwritten recognition.
The Best model used Gated convolution network as feature extractor and multi head attention layers
"""
import os
import cv2
import glob
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from .attention import *

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class flor(torch.nn.Module):
    def __init__(self):
        super(flor,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            GatedConv2dWithActivation(16,16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            GatedConv2dWithActivation(32,32, kernel_size=3, padding=1),
            nn.Conv2d(32, 40, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            GatedConv2dWithActivation(40,40, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(40, 48, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            GatedConv2dWithActivation(48,48, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, 56, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            GatedConv2dWithActivation(56,56, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(56, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
        )
        self.gru1 = nn.GRU(input_size=128,
                            hidden_size = 128,
                            num_layers=1,
                            dropout=0.5,
                            bidirectional=True)
        self.gru2 = nn.GRU(input_size=128,
                            hidden_size = 128,
                            num_layers=1,
                            dropout=0.5,
                            bidirectional=True)
        self.linear1 = nn.Linear(in_features=256,out_features=128)
        self.linear2 = nn.Linear(in_features=256,out_features=97)
    def forward(self,x):
        x = self.encoder(x)
        x = x.permute(0,3,1,2)
        x = x.flatten(2)
        x,h = self.gru1(x)
        x = self.linear1(x)
        x,h = self.gru2(x)
        print(x.size())
        logits = self.linear2(x)
        return logits

class flor_lstm(torch.nn.Module):
    """
    CNN encoder followed with lstm
    """
    def __init__(self):
        super(flor_lstm,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            GatedConv2dWithActivation(16,16, kernel_size=3, padding=1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            GatedConv2dWithActivation(32,32, kernel_size=3, padding=1),

            nn.Conv2d(32, 40, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(40),
            nn.PReLU(),
            GatedConv2dWithActivation(40,40, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(40, 48, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            GatedConv2dWithActivation(48,48, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(48, 56, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(56),
            nn.PReLU(),
            GatedConv2dWithActivation(56,56, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(56, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size = 128,
                            num_layers=4,
                            dropout=0.2,
                            bidirectional=True)
        self.linear = nn.Linear(in_features=256,out_features=97)

    def forward(self,x):
        x = self.encoder(x)
        x = x.permute(0,3,1,2)
        x = x.flatten(2)
        x,h = self.lstm(x)
        logits = self.linear(x)
        return logits

class flor_attention(torch.nn.Module):
    """
    Gated CNN encoder and followed with attention model
    """
    def __init__(self):
        super(flor_attention,self).__init__()
        self.config = config()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            GatedConv2dWithActivation(16,16, kernel_size=3, padding=1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            GatedConv2dWithActivation(32,32, kernel_size=3, padding=1),

            nn.Conv2d(32, 40, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(40),
            nn.PReLU(),
            GatedConv2dWithActivation(40,40, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(40, 48, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            GatedConv2dWithActivation(48,48, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(48, 56, kernel_size=(4,2), stride=(4,2)),
            nn.BatchNorm2d(56),
            nn.PReLU(),
            GatedConv2dWithActivation(56,56, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(56, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
        )
        self.atten = AlbertLayerGroup(self.config)
        self.linear = nn.Linear(in_features=128,out_features=97)

    def forward(self,x):
        x = self.encoder(x)
        x = x.permute(0,3,1,2)
        x = x.flatten(2)
        x = self.atten(x)
        logits = self.linear(x[0])
        return logits