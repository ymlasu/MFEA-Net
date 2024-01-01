import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import feanet.pac as pac

class DerivativeNet(nn.Module):
    def __init__(self, h, nmask, emask):
        super(DerivativeNet, self).__init__()
        self.nmask = nmask
        self.xNet= pac.PacPool2d(
            out_channels=1, kernel_size=(1,3), padding=1, kernel_type='dx')
        self.yNet = pac.PacPool2d(
            out_channels=1, kernel_size=(3,1), padding=1, kernel_type='dy')
        self.xK_kernels = 1/h*self.xNet.compute_kernel(emask) # (bs, 1, 1, 3, h_node, w_node)
        self.yK_kernels = 1/h*self.yNet.compute_kernel(emask) # (bs, 1, 3, 1, h_node, w_node)

    def forward(self, u, direction):
        if(direction == 'x'):
            mask = self.nmask.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 3, 1, 1) # (bs, 1, 1, 3, h_node, w_node)
            return self.xNet(u, None, self.xK_kernels*mask)
        if(direction == 'y'):
            mask = self.nmask.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 1, 1, 1) # (bs, 1, 3, 1, h_node, w_node)
            return self.yNet(u, None, self.yK_kernels*mask)