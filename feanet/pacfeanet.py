import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import feanet.pac as pac


class PACFEANet(nn.Module):
    def __init__(self, device, no_neumann, mode='thermal', kernel_size=3):
        super(PACFEANet, self).__init__()
        self.mode = mode
        self.device = device
        self.no_neumann = no_neumann # false of considering neumann boundary 
        self.h = 1.  # pixel size
        self.km, self.ku, self.kf = 1, 1, 1  # thermal problem
        if (self.mode != 'thermal'):
            self.km, self.ku, self.kf = 2, 2, 2

        self.K_kernels = None
        self.f_kernels = None
        self.materials = None
        self.t_kernels = [] # t_kernel should be a list, since there may be multiple neumann connections

        self.sac_Knet = pac.PacPool2d(
            out_channels=self.ku*self.kf, kernel_size=kernel_size, padding=1, kernel_type='quad')
        self.sac_fnet = pac.PacPool2d(
            out_channels=self.kf, kernel_size=kernel_size, padding=1, kernel_type='quad')  # body force
        self.sac_tnet = pac.PacPool1d(
            out_channels=self.kf, kernel_size=kernel_size, padding=1, kernel_type='linear')  # traction force

        self.group_Knet = nn.Conv2d(
            in_channels=self.ku*self.kf, out_channels=self.kf, kernel_size=1, groups=self.ku, bias=False)
        self.group_Knet.weight.data.fill_(1.0)

    def thermal_elements(self, m):
        # only one material parameter, alpha
        alpha = m

        k11 = 4./6.*alpha.contiguous()
        k12 = -1./6.*alpha.contiguous()
        k13 = -2./6.*alpha.contiguous()
        k14 = -1./6.*alpha.contiguous()

        K = torch.cat((k11.clone(), k12.clone(), k13.clone(), k14.clone(),
                       k12.clone(), k11.clone(), k14.clone(), k13.clone(),
                       k13.clone(), k14.clone(), k11.clone(), k12.clone(),
                       k14.clone(), k13.clone(), k12.clone(), k11.clone()), dim=1)
        return torch.unsqueeze(K, dim=2).to(self.device)

    def elastic_elements(self, m):
        '''See derivation in Mathematica notebook'''
        K11 = self.elastic_element11(m) # (bs, 16, h_elem, w_elem)
        K12 = self.elastic_element12(m)
        K21 = self.elastic_element21(m)
        K22 = self.elastic_element22(m)

        # create a new dimension
        return torch.stack((K11, K12, K21, K22), dim=2).to(self.device)

    def elastic_element11(self, m):
        # generate the layer of element stiffness matrix K11
        E = torch.unsqueeze(m[:, 0, :, :], dim=1)
        v = torch.unsqueeze(m[:, 1, :, :], dim=1)

        if(self.mode == 'elastic_pstress'):
            k11 = (E/16./(1-v*v)*(8.-8./3*v)).contiguous()
            k12 = (E/16./(1-v*v)*(-4.-4./3*v)).contiguous()
            k13 = (E/16./(1-v*v)*(-4.+4./3*v)).contiguous()
            k14 = (E/16./(1-v*v)*8./3*v).contiguous()
        elif(self.mode == 'elastic_pstrain'):
            k11 = (E/6./(-1+v+2*v*v)*(-3.+4*v)).contiguous()
            k12 = (E/12./(-1+v+2*v*v)*(3.-2*v)).contiguous()
            k13 = (E/12./(-1+v+2*v*v)*(3.-4*v)).contiguous()
            k14 = (-E/6./(-1+v+2*v*v)*v).contiguous()

        K11 = torch.cat((k11.clone(), k12.clone(), k13.clone(), k14.clone(),
                         k12.clone(), k11.clone(), k14.clone(), k13.clone(),
                         k13.clone(), k14.clone(), k11.clone(), k12.clone(),
                         k14.clone(), k13.clone(), k12.clone(), k11.clone()), dim=1)  # cat in old dimension
        return K11

    def elastic_element12(self, m):
        # generate the layer of element stiffness matrix K12
        E = torch.unsqueeze(m[:, 0, :, :], dim=1)
        v = torch.unsqueeze(m[:, 1, :, :], dim=1)

        if(self.mode == 'elastic_pstress'):
            k11 = (E/8./(1-v)).contiguous()
            k12 = (E*(1-3*v)/8./(-1+v*v)).contiguous()
        elif(self.mode == 'elastic_pstrain'):
            k11 = (-E/8./(-1+v+2*v*v)).contiguous()
            k12 = (E*(1-4*v)/8./(-1+v+2*v*v)).contiguous()

        K12 = torch.cat((k11.clone(), k12.clone(), -k11.clone(), -k12.clone(),
                         -k12.clone(), -k11.clone(), k12.clone(), k11.clone(),
                         -k11.clone(), -k12.clone(), k11.clone(), k12.clone(),
                         k12.clone(), k11.clone(), -k12.clone(), -k11.clone()), dim=1)
        return K12

    def elastic_element21(self, m):
        # generate the layer of element stiffness matrix K21
        E = torch.unsqueeze(m[:, 0, :, :], dim=1)
        v = torch.unsqueeze(m[:, 1, :, :], dim=1)

        if(self.mode == 'elastic_pstress'):
            k11 = (E/8./(1-v)).contiguous()
            k12 = (E*(1-3*v)/8./(1-v*v)).contiguous()
        elif(self.mode == 'elastic_pstrain'):
            k11 = (-E/8./(-1+v+2*v*v)).contiguous()
            k12 = (E*(-1+4*v)/8./(-1+v+2*v*v)).contiguous()

        K21 = torch.cat((k11.clone(), k12.clone(), -k11.clone(), -k12.clone(),
                         -k12.clone(), -k11.clone(), k12.clone(), k11.clone(),
                         -k11.clone(), -k12.clone(), k11.clone(), k12.clone(),
                         k12.clone(), k11.clone(), -k12.clone(), -k11.clone()), dim=1)

        return K21

    def elastic_element22(self, m):
        # generate the layer of element stiffness matrix K22
        E = torch.unsqueeze(m[:, 0, :, :], dim=1)
        v = torch.unsqueeze(m[:, 1, :, :], dim=1)

        if(self.mode == 'elastic_pstress'):
            k11 = (E*(-3.+v)/6./(-1+v*v)).contiguous()
            k12 = (-E*v/6./(-1+v*v)).contiguous()
            k13 = (E*(3.-v)/12./(-1+v*v)).contiguous()
            k14 = (E*(3.+v)/12./(-1+v*v)).contiguous()
        elif(self.mode == 'elastic_pstrain'):
            k11 = (E/6./(-1+v+2*v*v)*(-3.+4*v)).contiguous()
            k12 = (-E/6./(-1+v+2*v*v)*v).contiguous()
            k13 = (E/12./(-1+v+2*v*v)*(3.-4*v)).contiguous()
            k14 = (E/12./(-1+v+2*v*v)*(3.-2*v)).contiguous()
            
        K22 = torch.cat((k11.clone(), k12.clone(), k13.clone(), k14.clone(),
                         k12.clone(), k11.clone(), k14.clone(), k13.clone(),
                         k13.clone(), k14.clone(), k11.clone(), k12.clone(),
                         k14.clone(), k13.clone(), k12.clone(), k11.clone()), dim=1)
        return K22

    def bodyforce_element(self, m):
        bs, _, hs, ws = m.shape
        el = torch.zeros(size=(bs, 16, self.kf, hs, ws)).to(self.device)
        el[:, 0, :, :, :], el[:, 1, :, :, :], el[:, 2, :, :, :], el[:, 3, :, :, :] = 4/9, 2/9, 1/9, 2/9
        el[:, 4, :, :, :], el[:, 5, :, :, :], el[:, 6, :, :, :], el[:, 7, :, :, :] = 2/9, 4/9, 2/9, 1/9
        el[:, 8, :, :, :], el[:, 9, :, :, :], el[:, 10, :, :, :], el[:, 11, :, :, :] = 1/9, 2/9, 4/9, 2/9
        el[:, 12, :, :, :], el[:, 13, :, :, :], el[:, 14, :, :, :], el[:, 15, :, :, :] = 2/9, 1/9, 2/9, 4/9
        return self.h*self.h/4.*el

    def traction_element(self, t_conn):
        # create 1-D element (input t_conn is a 2D array with the shape (bs,h,w))
        bs = t_conn.shape[0]
        conn_len = torch.where(t_conn[0].reshape(-1) == -1)[0][0]
        el = torch.zeros(size=(bs, 4, self.kf, conn_len-1)).to(self.device)
        el[:, 0, :, :], el[:, 1, :, :] = 2/3, 1/3
        el[:, 2, :, :], el[:, 3, :, :] = 1/3, 2/3
        return self.h/2.*el

    def input_clone(self, u, kf):
        # clone kf inputs in dim=1 dimension
        initial_u = u.clone()
        for _ in range(kf-1):
            u = torch.cat((u, initial_u), dim=1)
        return u

    def calc_KU(self, u, m, nmsk):
        if ((self.K_kernels == None) or (not torch.equal(m, self.materials))):
            stiffness = self.thermal_elements(m)
            if (self.mode == 'elastic_pstress' or self.mode == 'elastic_pstrain'):
                stiffness = self.elastic_elements(m) # (bs, 16, 4, h_elem, w_elem)
            self.K_kernels = self.sac_Knet.compute_kernel(stiffness) # (bs, ks, 3, 3, h_node, w_node)
            self.mask = nmsk.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3, 1, 1) # (bs, ku*kf, 3, 3, h_node, w_node)
            self.materials = m

        u_clone = self.input_clone(u, self.kf)
        f_sac = self.sac_Knet(u_clone, None, self.K_kernels*self.mask)
        return self.group_Knet(f_sac)

    def calc_bodyforce(self, h, f, m):
        # bodyforce is related to mesh size
        if ((self.f_kernels == None) or (h != self.h)):
            self.h = h
            bodyforce_elements = self.bodyforce_element(m)
            self.f_kernels = self.sac_fnet.compute_kernel(bodyforce_elements)
        return self.sac_fnet(f, None, self.f_kernels*self.mask)
        
    def calc_neumannbc(self, h, t, t_conn):
        # t_conn has shape of (bs, n_conn, h, w), n_conn is the number of connection lists
        # t channel will be 1*n_conn for thermal problems; 2*n_conn for elastic problems
        bs, n_conn, hs, ws = t_conn.shape
        if ((self.t_kernels == []) or (h != self.h)):
            self.h = h
            for c in range(n_conn):
                traction_elements = self.traction_element(t_conn[:,c,:,:])
                t_kernels, _ = self.sac_tnet.compute_kernel(traction_elements) # output (bs, ks, 3, l)
                self.t_kernels.append(t_kernels)

        # stack all the neumann bc together
        t_baseline = torch.zeros(bs,self.kf,hs,ws)
        for b in range(bs):
            for c in range(n_conn):
                t_conn_flat = t_conn[b,c,:,:].reshape(-1)
                conn = t_conn_flat[t_conn_flat != -1]
                neumann_node = conn.unsqueeze(0).unsqueeze(0).expand(1, self.kf, conn.shape[0])
                # get the 1-D neumann boundary value
                t_flat = torch.flip(t[b,self.kf*c:self.kf*(c+1)].unsqueeze(0), dims=[2]).view(1,self.kf,hs*ws)
                t_neumann = t_flat.gather(2, neumann_node) # find element in t_flat for dim=2
                # perform 1-D convolution
                t_conv = self.sac_tnet(t_neumann, None, self.t_kernels[c])
                # convert back to neumann boundary map, zero it first
                t_flat.zero_()
                t_flat.scatter_(-1, neumann_node, t_conv) # scatter is the inverse function of gather
                t_baseline[b] = t_baseline[b] + torch.flip(t_flat.view(self.kf,hs,ws), dims=[1])
        return t_baseline

    def calc_F(self, h, f, t, t_conn, m):
        if (self.no_neumann):
            return self.calc_bodyforce(h, f, m)
        else:
            return self.calc_bodyforce(h, f, m)+self.calc_neumannbc(h, t, t_conn)

    def forward(self, term_KU=None, term_F=None, h=None, u=None, d_idx=None, f=None, t=None, t_conn=None, m=None, nmsk=None):
        # for elasticity problems, m(material_input) has two channels, E and v
        # h is pixel size
        # u is initial solution
        # f is body force of unit volume, with kf channels
        # t is traction map, with kf channels
        self.term_KU = term_KU
        if (term_KU == None):
            self.term_KU = self.calc_KU(u, m, nmsk)

        self.term_F = term_F
        if (term_F == None):
            self.term_F = self.calc_F(h, f, t, t_conn, m)

        return (self.term_F-self.term_KU)*d_idx

        '''
        self.u_clone = self.input_clone(u, self.kf)
        self.f_sac = self.sac_Knet(self.u_clone, None, self.K_kernels)

        self.temp1 = self.group_Knet(self.f_sac)
        self.temp2 = self.sac_fnet(f, None, self.f_kernels)
        self.temp3 = self.sac_tnet(t, None, self.t_kernels)*t_idx
        return self.temp2+self.temp3-self.temp1
        '''
