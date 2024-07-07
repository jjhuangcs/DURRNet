import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class ResBlock(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResBlock, self).__init__()

        if_bias = True
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_out', nonlinearity='relu')
        self.conv2 = nn.Conv2d(in_channels=f_ch, out_channels=in_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_out', nonlinearity='relu')

        self.relu = nn.LeakyReLU() # ReLU

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))


class PUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f_ch, f_sz, num_layers, dilate):
        super(PUNet, self).__init__()

        if_bias = False

        self.layers = []
        self.layers.append(nn.Conv2d(in_ch, f_ch, f_sz, stride=1, padding=math.floor(f_sz / 2), bias=False))
        self.layers.append(nn.ReLU())
        for i in range(num_layers):
            self.layers.append(ResBlock(in_ch=f_ch, f_ch=f_ch, f_sz=f_sz, dilate=1))
        self.net = nn.Sequential(*self.layers)

        self.convOut = nn.Conv2d(f_ch, out_ch, f_sz, stride=1, padding=math.floor(f_sz / 2) + dilate - 1,
                                 dilation=dilate, bias=if_bias)
        self.convOut.weight.data.fill_(0.)

    def forward(self, x):
        x = self.net(x)
        out = self.convOut(x)
        return out


class LiftingStep(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):
        super(LiftingStep, self).__init__()

        self.dilate = dilate

        pf_ch = int(f_ch)
        uf_ch = int(f_ch)
        self.predictor = PUNet(pin_ch, uin_ch, pf_ch, f_sz, num_layers, dilate)
        self.updator = PUNet(uin_ch, pin_ch, uf_ch, f_sz, num_layers, dilate)

    def forward(self, xc, xd):
        Fxc = self.predictor(xc)
        xd = - Fxc + xd
        Fxd = self.updator(xd)
        xc = xc + Fxd

        return xc, xd

    def inverse(self, xc, xd):
        Fxd = self.updator(xd)
        xc = xc - Fxd
        Fxc = self.predictor(xc)
        xd = xd + Fxc

        return xc, xd

class invNet(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_step, num_layers):
        super(invNet, self).__init__()
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LiftingStep(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
        self.net = mySequential(*self.layers)

    def forward(self, xc, xd):
        for i in range(len(self.net)):
            xc, xd = self.net[i].forward(xc, xd)
        return xc, xd

    def inverse(self, xc, xd):
        for i in reversed(range(len(self.net))):
            xc, xd = self.net[i].inverse(xc, xd)
        return xc, xd