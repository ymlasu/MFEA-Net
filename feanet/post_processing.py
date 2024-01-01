import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from feanet.derivative import DerivativeNet

def ConvertNodeMaterial(m, emask):
    '''Convert the element-based material field into node-based ones'''
    m_padded = F.pad(m*emask, (1, 1, 1, 1), mode='constant', value=0)
    emask_padded = F.pad(emask, (1, 1, 1, 1), mode='constant', value=0)
    kernel = torch.ones((1, 1, 2, 2), dtype=m.dtype, device=m.device) # create a kernel for convolution

    # Perform convolution to sum adjacent elements
    conv_sum = F.conv2d(m_padded, kernel, padding=0, groups=m.shape[1])

    # Convolve the mask to get the count of valid adjacent elements
    count = F.conv2d(emask_padded, kernel, padding=0, groups=emask.shape[1])

    return conv_sum / count.clamp(min=1) # Calculate the average


class ThermalPostProcessing(nn.Module):
    def __init__(self, h, m, emask, nmask):
        super(ThermalPostProcessing, self).__init__()
        self.kf = 1
        self.emask = emask
        self.flux = None
        self.m_node = ConvertNodeMaterial(m, emask) # (bs, 1, h+1, w+1)
        self.deriv = DerivativeNet(h, self.kf, nmask)

    def ComputeFlux(self, u):
        B, _, H, W = u.shape
        self.flux = torch.zeros((B,2,H,W)).double()
        self.flux[:,0,:,:] = self.deriv(u, 'x')
        self.flux[:,1,:,:] = self.deriv(u, 'y')
        self.flux *= -self.m_node
        return self.flux


class ElasticPostProcessing(nn.Module):
    def __init__(self, h, mode, m, emask, nmask):
        super(ElasticPostProcessing, self).__init__()
        stiffness = self.ComputeStiffnessMatrix(mode, m) # (bs, 6, h, w)
        self.kf = 2
        self.strain = None
        self.stress = None
        self.emask = emask
        self.m_node = ConvertNodeMaterial(stiffness, emask) # (bs, 6, h+1, w+1)
        self.deriv = DerivativeNet(h, self.kf, nmask)
        
    def ComputeStiffnessMatrix(self, mode, m):
        '''Compute stiffness matrix using the E, v field'''
        E = torch.unsqueeze(m[:, 0, :, :], dim=1)
        v = torch.unsqueeze(m[:, 1, :, :], dim=1)
        if(mode == 'elastic_pstress'):
            d11 = (E/(1-v*v)).contiguous()
            d12 = (E*v/(1-v*v)).contiguous()
            d33 = (E/2/(1+v)).contiguous()
        else:
            d11 = (E*(1-v)/(1+v)/(1-2*v)).contiguous()
            d12 = (E*v/(1+v)/(1-2*v)).contiguous()
            d33 = (E/2/(1+v)).contiguous()
        return torch.cat((d11.clone(), d12.clone(), d33.clone()), dim=1)


    def ComputeStrain(self, u):
        B, _, H, W = u.shape
        self.strain = torch.zeros((B,3,H,W)).double()
    
    def ComputeStress(self, u):
        self.ComputeStrain(u)