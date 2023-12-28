import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DerivativeNet(nn.Module):
    def __init__(self, chs, h):
        super(DerivativeNet, self).__init__()
        self.xNet_internal = nn.Conv2d(chs, chs, kernel_size=(1,3), padding=(0,1), bias=False, groups=chs)
        self.xNet_edge = nn.Conv2d(chs, chs, kernel_size=(1,2), padding=0, bias=False, groups=chs)
        self.yNet_internal = nn.Conv2d(chs, chs, kernel_size=(3,1), padding=(1,0), bias=False, groups=chs)
        self.yNet_edge = nn.Conv2d(chs, chs, kernel_size=(2,1), padding=0, bias=False, groups=chs)
        
        self.xNet_internal.requires_grad_(False)
        self.xNet_edge.requires_grad_(False)
        self.yNet_internal.requires_grad_(False)
        self.yNet_edge.requires_grad_(False)

        for i in range(chs):
            self.xNet_internal.state_dict()['weight'][i][0] = torch.asarray([-1., 0., 1.]) / (2.0 * h)
            self.xNet_edge.state_dict()['weight'][i][0] = torch.asarray([-1., 1.]) / h
            self.yNet_internal.state_dict()['weight'][i][0] = torch.asarray([[1.], [0.], [-1.]]) / (2.0 * h)
            self.yNet_edge.state_dict()['weight'][i][0] = torch.asarray([[1.], [-1.]]) / h

    def forward(self, u, msk, direction):
        if(direction == 'x'):
            derivative = self.xNet_internal(u)
            edge = self.xNet_edge(u)
            derivative[:, :, :, 0] = edge[:, :, :, 0]
            derivative[:, :, :, -1] = edge[:, :, :, -1]
        if(direction == 'y'):
            derivative = self.yNet_internal(u)
            edge = self.yNet_edge(u)
            derivative[:, :, 0, :] = edge[:, :, 0, :]
            derivative[:, :, -1, :] = edge[:, :, -1, :]
        return derivative