import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DerivativeNet(nn.Module):
    def __init__(self, chs, h):
        super(DerivativeNet, self).__init__()
        self.xNet_internal = nn.Conv2d(chs, chs, kernel_size=(1,3), padding=(0,1), bias=False, groups=chs)
        self.xNet_left = nn.Conv2d(chs, chs, kernel_size=(1,3), padding=(0,1), bias=False, groups=chs)
        self.xNet_right = nn.Conv2d(chs, chs, kernel_size=(1,3), padding=(0,1), bias=False, groups=chs)
        self.yNet_internal = nn.Conv2d(chs, chs, kernel_size=(3,1), padding=(1,0), bias=False, groups=chs)
        self.yNet_top = nn.Conv2d(chs, chs, kernel_size=(3,1), padding=(1,0), bias=False, groups=chs)
        self.yNet_bottom = nn.Conv2d(chs, chs, kernel_size=(3,1), padding=(1,0), bias=False, groups=chs)
        
        for i in range(chs):
            self.xNet_internal.state_dict()['weight'][i][0] = torch.asarray([-1., 0., 1.]) / (2.0 * h)
            self.xNet_left.state_dict()['weight'][i][0] = torch.asarray([0., -1., 1.]) / h
            self.xNet_right.state_dict()['weight'][i][0] = torch.asarray([-1., 1., 0.]) / h
            self.yNet_internal.state_dict()['weight'][i][0] = torch.asarray([[1.], [0.], [-1.]]) / (2.0 * h)
            self.yNet_top.state_dict()['weight'][i][0] = torch.asarray([[0.], [1.], [-1.]]) / h
            self.yNet_bottom.state_dict()['weight'][i][0] = torch.asarray([[1.], [-1.], [0.]]) / h

        self.xNet_internal.requires_grad_(False)
        self.xNet_left.requires_grad_(False)
        self.xNet_right.requires_grad_(False)
        self.yNet_internal.requires_grad_(False)
        self.yNet_top.requires_grad_(False)
        self.yNet_bottom.requires_grad_(False)


    def shrink_mask(self, msk, direction):
        if(direction == 'x'):
            kernel = torch.ones((1, 1, 1, 3), dtype=torch.float32)
            conv_result = F.conv2d(msk, kernel, padding=(0,1))
            shrink_mask = (conv_result == 3).float() #(ensure all three pixels were 'on')
        if(direction == 'y'):
            kernel = torch.ones((1, 1, 3, 1), dtype=torch.float32)
            conv_result = F.conv2d(msk, kernel, padding=(1,0))
            shrink_mask = (conv_result == 3).float() #(ensure all three pixels were 'on')
        return shrink_mask

    def eroded_pixels(self, mask, direction):
        if(direction == 'x'):
            cumsum = torch.cumsum(mask, dim=-1) # the cumulative sum along the width dimension
            edge1 = (cumsum == 1) # positions where cumsum changes from 0 to a positive value
            row_max = cumsum.max(dim=-1, keepdim=True).values
            edge2 = (cumsum == row_max) & mask
        if(direction == 'y'):
            cumsum = torch.cumsum(mask, dim=-2) # the cumulative sum along the height dimension
            edge1 = (cumsum == 1) # positions where cumsum changes from 0 to a positive value
            col_max = cumsum.max(dim=-2, keepdim=True).values
            edge2 = (cumsum == col_max) & mask
        return edge1, edge2

    def forward(self, u, mask, direction):
        '''
        edge_mask = mask - eroded_mask
        1- compute the internal derivative
        2- compute the edge derivative
        3- extract the internal derivative using eroded_mask
        4- extract the edge derivative using edge_mask
        5- add the internal and edge derivative together
        '''
        eroded_mask = self.shrink_mask(mask, direction)
        edge1_mask, edge2_mask = self.eroded_pixels(mask.bool(), direction)

        if(direction == 'x'):
            internal_d = eroded_mask * self.xNet_internal(u)
            left_d = edge1_mask * self.xNet_left(u)
            right_d = edge2_mask * self.xNet_right(u)
            return internal_d + left_d + right_d
        if(direction == 'y'):
            internal_d = eroded_mask * self.yNet_internal(u)
            top_d = edge1_mask * self.yNet_top(u)
            bottom_d = edge2_mask * self.yNet_bottom(u)
            return internal_d + top_d + bottom_d