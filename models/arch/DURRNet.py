import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import numpy as np
import pywt
import torch.nn.functional as F
from invNet import *

def softshrink(x, lambd):
    sgn = torch.sign(x)
    tmp = torch.abs(x)-lambd
    out = sgn*(tmp + torch.abs(tmp))/2
    return out

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


class waveletDecomp(nn.Module):
    def __init__(self, stride=2):
        super(waveletDecomp, self).__init__()

        self.stride = stride

        wavelet = pywt.Wavelet('haar')  # haar bior3.3
        dec_hi = torch.tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.tensor(wavelet.dec_lo[::-1])

        self.filters_dec = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_dec = self.filters_dec.unsqueeze(1)
        self.psize = int(self.filters_dec.size(3) / 2)

        rec_hi = torch.tensor(wavelet.rec_hi[::-1])
        rec_lo = torch.tensor(wavelet.rec_lo[::-1])
        self.filters_rec = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_rec = self.filters_rec.unsqueeze(0)

        self.filters_rec_transposed = torch.flip(self.filters_rec.permute(1, 0, 2, 3), [2, 3])

    def forward(self, x):
        chals = x.size(1)
        out = []
        if self.stride == 1:
            x = F.pad(x, (self.psize - 1, self.psize, self.psize - 1, self.psize), mode='replicate')

        for i in range(chals):
            coeff = F.conv2d(x[:, i, :, :].unsqueeze(1), self.filters_dec, stride=self.stride,
                                         bias=None, padding=0)

            if i == 0:
                out = coeff
            else:
                out = torch.cat((out, coeff), 1)

        return out / 2

    def inverse(self, x):
        chals = x.size(1)
        out = []
        if self.stride == 1:
            x = F.pad(x, (self.psize, self.psize - 1, self.psize, self.psize - 1), mode='replicate')

        for i in range(int(chals / 4)):
            if self.stride == 1:
                coeff = F.conv2d(x[:, i*4:(i+1)*4, :, :], self.filters_rec, stride=self.stride, bias=None, padding=0)
            else:
                coeff = F.conv_transpose2d(x[:, i*4:(i+1)*4, :, :], self.filters_rec_transposed, stride=self.stride,
                                                           bias=None, padding=0)
            if i == 0:
                out = coeff
            else:
                out = torch.cat((out, coeff), 1)

        return (out * self.stride ** 2) / 2


class unfoldseplayer_inv_excl(nn.Module):
    def __init__(self, fxin, fsz, chals):
        super(unfoldseplayer_inv_excl, self).__init__()
        self.chals = chals

        self.convDTf = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.convDf = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)

        self.convDTb = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
        self.convDb = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)

        self.proxNetlayers = 1

        self.scale = 1
        self.exclNetlayers = 1
        self.steps = 2

        self.softplus = nn.Softplus(beta=20)

        self.tauf = Parameter(1 * torch.ones(1), requires_grad=True)
        self.taub = Parameter(1 * torch.ones(1), requires_grad=True)

        self.etaf = Parameter(0.5 * torch.ones(1), requires_grad=True)
        self.etab = Parameter(0.5 * torch.ones(1), requires_grad=True)
        # proxNet
        self.PNetf = proxyNetv0(nlayer=self.proxNetlayers, nchal=int(chals), fsz=3)
        self.PNetb = proxyNetv0(nlayer=self.proxNetlayers, nchal=int(chals), fsz=3)
        # proxInvNet
        self.ExclNetf = proxInvExclThreNetms(scale=self.scale, fsz=3, fch=chals, fxin=fxin, nsteps=self.steps, nlayers=self.exclNetlayers)
        self.ExclNetb = proxInvExclThreNetms(scale=self.scale, fsz=3, fch=chals, fxin=fxin, nsteps=self.steps, nlayers=self.exclNetlayers)

    def forward(self, x, T, R, zT, zR, if_last):

        zT = zT + self.convDTf(T - self.convDf(zT))
        zR = zR + self.convDTb(R - self.convDb(zR))


        zT = self.PNetf(zT)
        zR = self.PNetb(zR)
        ######################################### Estimate Reflection Layer ############################################
        R = R + self.softplus(self.etab) * ((x - T - R) - self.softplus(self.taub) * (R - self.convDb(zR)))
        R = self.ExclNetb(R, T)
        ################################################################################################################

        ########################################## Estimation Transmission Layer #######################################
        T = T + self.softplus(self.etaf) * ((x - T - R) - self.softplus(self.tauf) * (T - self.convDf(zT)))
        T = self.ExclNetf(T, R)
        ################################################################################################################

        if if_last == 1:
            zT = self.convDf(zT)
            zR = self.convDb(zR)

        # T is the transmission image, R is the reflection image
        return T, R, zT, zR

class proxyNetv0(nn.Module):
    def __init__(self, nlayer, nchal, fsz):
        super(proxyNetv0, self).__init__()
        layers = []
        layers.append(nn.Conv2d(nchal, nchal, fsz, stride=1, padding=math.floor(fsz / 2)))
        layers.append(nn.ReLU())
        for ii in range(nlayer):
            layers.append(ResBlock(in_ch=nchal, f_ch=nchal, f_sz=fsz))
        layers.append(nn.Conv2d(nchal, nchal, fsz, stride=1, padding=math.floor(fsz / 2)))
        self.PNet = nn.Sequential(*layers)

    def forward(self, x):
        out = self.PNet(x) + x
        return out

class proxInvExclThreNetms(nn.Module):
    def __init__(self, scale, fsz, fxin, fch, nsteps, nlayers):
        super(proxInvExclThreNetms, self).__init__()

        self.scale = scale
        self.fxin = fxin

        self.wavelet = waveletDecomp(stride=2)

        self.invNetlayer = []
        for i in range(self.scale):
            self.invNetlayer.append(invNet(pin_ch=fxin, f_ch=fch, uin_ch=fxin * 3, f_sz=fsz,
                                           dilate=1, num_step=nsteps, num_layers=nlayers))
        self.invNet = mySequential(*self.invNetlayer)

        self.threNetlayer = []
        for i in range(self.scale):
            self.threNetlayer.append(PUNet(in_ch=fxin * 3 * 2, out_ch=fxin * 3, f_ch=fch, f_sz=fsz, num_layers=1, dilate=1))
        self.threNetlayer.append(PUNet(in_ch=fxin * 1 * 2, out_ch=fxin * 1, f_ch=fch, f_sz=fsz, num_layers=1, dilate=1))
        self.threNet = mySequential(*self.threNetlayer)

    def forward(self, x1, x2):
        n1 = torch.norm(x1)
        n2 = torch.norm(x2)
        x2 = (n1 / n2) * x2

        xd1, xd2, xc = [], [], []
        x1c, x2c = x1, x2
        for i in range(self.scale):
            coeff1 = self.wavelet.forward(x1c)
            coeff2 = self.wavelet.forward(x2c)

            x1c, x1d = waveletcoeffsplit(coeff1, self.fxin)
            x1c, x1d = self.invNet[i].forward(x1c, x1d)

            x2c, x2d = waveletcoeffsplit(coeff2, self.fxin)
            x2c, x2d = self.invNet[i].forward(x2c, x2d)

            '''Thresholding Network'''
            x1d = x1d + self.threNet[i](torch.cat((x1d, x2d), 1))

            xd1.append(x1d)

            if i == self.scale - 1:
                '''Thresholding Network'''
                x1c = x1c + self.threNet[i+1](torch.cat((x1c, x2c), 1))

        for i in reversed(range(self.scale)):
            x1c, x1d = self.invNet[i].inverse(x1c, xd1[i])
            coeff1 = waveletcoeffmerge(x1c, x1d, self.fxin)
            x1c = self.wavelet.inverse(coeff1)

        if x1c.size() != x1.size():
            x1c = nn.functional.upsample(x1c, size=[x1.size(2), x1.size(3)], mode='bilinear')
        return x1c


def waveletcoeffsplit(coeff, fxin):
    for i in range(fxin):
        if i == 0:
            xc = coeff[:, i * 4:i * 4 + 1, :, :]
            xd = coeff[:, i * 4 + 1:(i + 1) * 4, :, :]
        else:
            xc = torch.cat((xc, coeff[:, i * 4:i * 4 + 1, :, :]), 1)
            xd = torch.cat((xd, coeff[:, i * 4 + 1:(i+1) * 4, :, :]), 1)
    return xc, xd


def waveletcoeffmerge(xc, xd, fxin):
    for i in range(fxin):
        if i == 0:
            coeff = xc[:, i, :, :].unsqueeze(1)
            coeff = torch.cat((coeff, xd[:, i * 3:(i + 1) * 3, :, :]), 1)
        else:
            coeff = torch.cat((coeff, xc[:, i, :, :].unsqueeze(1)), 1)
            coeff = torch.cat((coeff, xd[:, i * 3:(i+1) * 3, :, :]), 1)
    return coeff

class DCN(nn.Module):
    def __init__(self, in_chals, out_chals, layers, chals, fsz0, fsz):
        super(DCN, self).__init__()

        self.chals = chals
        self.layers = layers
        self.fsz = fsz

        layers = []
        for ii in range(self.layers):
            if ii == 0:
                layers.append(nn.Conv2d(in_chals, self.chals, fsz0, stride=1, padding=math.floor(fsz0 / 2), bias=True))
                layers.append(nn.ReLU(True))
            else:
                layers.append(
                    nn.Conv2d(self.chals, self.chals, self.fsz, stride=1, padding=math.floor(self.fsz / 2),
                              bias=True))
                layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(self.chals, out_chals, self.fsz, stride=1, padding=math.floor(self.fsz / 2)))
        self.DCNet = nn.Sequential(*layers)

    def forward(self, x):
        out = self.DCNet(x)
        return out


class DUSepNet(nn.Module):
    def __init__(self, n_channels):
        super(DUSepNet, self).__init__()
        fxin = 3

        self.chals = 64#
        self.layers = 2#
        self.fsz = 5#
        self.FEfsz = 3
        self.rlayers = 3
        self.scale = 4

        layers_net = []
        for _ in range(self.layers * self.scale):
            layers_net.append(unfoldseplayer_inv_excl(fxin=fxin, fsz=self.fsz, chals=self.chals))
        self.DUSepNet = mySequential(*layers_net)

        layers_FEzT, layers_FEzR = [], []
        for _ in range(self.scale):
            layers_FEzT.append(DCN(in_chals=n_channels, out_chals=self.chals, layers=self.rlayers, chals=64, fsz0=1, fsz=self.FEfsz))
            layers_FEzR.append(DCN(in_chals=n_channels, out_chals=self.chals, layers=self.rlayers, chals=64, fsz0=1, fsz=self.FEfsz))
        self.FEzT = mySequential(*layers_FEzT)
        self.FEzR = mySequential(*layers_FEzR)

        layerzT1x1, layerzR1x1 = [], []
        for _ in range(self.scale-1):
            layerzT1x1.append(nn.Conv2d(2 * self.chals, self.chals, 1, stride=1, padding=math.floor(1 / 2), bias=False))
            layerzR1x1.append(nn.Conv2d(2 * self.chals, self.chals, 1, stride=1, padding=math.floor(1 / 2), bias=False))
        self.zT1x1 = mySequential(*layerzT1x1)
        self.zR1x1 = mySequential(*layerzR1x1)

        self.scale_factor = 2

        self.up = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')#scale_factor=2
        self.down = nn.Upsample(scale_factor=1./self.scale_factor, mode='bilinear')#scale_factor=0.5

    def forward(self, x, y=None, fn=None):
        out_l, out_r = [], []
        if y is None:
            y = x

        yd = []
        xd = x
        yd.append(y)
        for s in range(self.scale-1):
            xd = self.down(xd)
            yd.append(self.down(yd[s]))
        '''0.5x'''
        T, R = 0.5 * xd, 0.5 * xd

        for s in range(self.scale):
            xd = x
            for _ in range(self.scale - s - 1):
                xd = self.down(xd)

            fzT = self.FEzT[s](yd[-s - 1])
            fzR = self.FEzR[s](yd[-s - 1])
            if s == 0:
                zT, zR = fzT, fzR
            else:
                zT = self.zT1x1[s - 1](torch.cat((zT, fzT), 1))
                zR = self.zR1x1[s - 1](torch.cat((zR, fzR), 1))

            for i in range(self.layers * s, self.layers * (s + 1)):
                T, R, zT, zR = self.DUSepNet[i](xd, T, R, zT, zR, 1 if (i == self.layers * (s + 1)-1) & (s == self.scale - 1) else 0)

            if s < self.scale - 1:
                if [self.scale_factor*T.size(-2), self.scale_factor*T.size(-1)] != [yd[-s-2].size(-2), yd[-s-2].size(-1)]:
                    T = nn.functional.upsample(T, size=[yd[-s-2].size(2), yd[-s-2].size(3)], mode='bilinear')
                    R = nn.functional.upsample(R, size=[yd[-s-2].size(2), yd[-s-2].size(3)], mode='bilinear')
                    zT = nn.functional.upsample(zT, size=[yd[-s-2].size(2), yd[-s-2].size(3)], mode='bilinear')
                    zR = nn.functional.upsample(zR, size=[yd[-s-2].size(2), yd[-s-2].size(3)], mode='bilinear')
                else:
                    T  = self.up(T)
                    R  = self.up(R)
                    zT = self.up(zT)
                    zR = self.up(zR)

        out_l.append(zT)
        out_l.append(T)

        out_r.append(zR)
        out_r.append(R)

        return out_l, out_r

