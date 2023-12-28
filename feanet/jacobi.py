import torch
import numpy as np
import torch.nn.functional as F

class JacobiBlock():
    """ Define all the methods necessary for a CNN-based Jacobi iteration; a Jacobi block must be defined with a single/batch of problems
        
        FEANet: neural network model to output residual
        h : pixel size
    """
    def __init__(self, net, h, device, mode='thermal'):
        self.device = device
        self.net = net # initialize the nn network
        self.mode = mode
        self.kf = 1 # thermal problem
        if(self.mode != 'thermal'):
            self.kf = 2

        self.omega = 2/3.
        self.h = h
        self.d_mat = None

    def reset_boundary(self, u, dirich_value, dirich_idx):
        """ Reset values at the dirichlet boundary """
        return u * dirich_idx + dirich_value

    def compute_diagonal_matrix(self):
        """ Comopute diagonal matrix for Jacobi iteration """
        if(self.mode == 'thermal'):
            d_mat = torch.unsqueeze(self.net.K_kernels[:, 0, 1, 1, :, :], dim=1) # pac K_kernels
        elif(self.mode == 'elastic_pstress' or self.mode == 'elastic_pstrain'):
            dxx = self.net.K_kernels[:, 0, 1, 1, :, :] 
            dyy = self.net.K_kernels[:, 3, 1, 1, :, :] 
            d_mat = torch.stack((dxx, dyy), dim=1)
        
        d_mat[d_mat.abs()<1e-6] = 1.0
        return d_mat

    def jacobi_convolution(self, u, m, msk, d, d_idx, term_KU=None, term_F=None, h=None, f=None, t=None, t_conn=None):
        """ 
        Jacobi method iteration step defined as a convolution:
        u_new = omega/d_mat*residual + u 
        d_idx : dirichlet boundary index
            Matrix describing the domain: 1.0 for inner points 0.0 elsewhere.
        d: dirichlet boundary value
            Matrix describing the domain: desired values for boundary points 0.0 elsewhere.
        """
        if(self.d_mat == None):
            self.net(term_KU, term_F, h, u, d_idx, f, t, t_conn, m, msk)
            self.d_mat = self.compute_diagonal_matrix().to(self.device)

        u = self.reset_boundary(u, d, d_idx)
        residual = self.net(term_KU, term_F, h, u, d_idx, f, t, t_conn, m, msk)
        u_new = self.omega/self.d_mat*residual + u
        return msk*self.reset_boundary(u_new, d, d_idx)