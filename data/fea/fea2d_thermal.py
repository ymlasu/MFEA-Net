import numpy as np
import random
import matplotlib.pyplot as plt
import fea.gaussian_random_fields as gr

class ThermalFEM():
    '''
    Note: input the mesh grid, material, bc and f values, generate a finite element problem
    '''
    def __init__(self, grid, material, dirich_val, neumann_val, f_val):
        self.grid = grid
        self.res_arr = []
        self.qpts = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])/np.sqrt(3) #[2x4], integration points
        self.n_nodes = grid.points.shape[0]
        self.d = np.zeros(self.n_nodes, dtype=np.float32)
        self.f = np.zeros_like(f_val, dtype=np.float32)
        self.flux = np.zeros((self.n_nodes,2), dtype=np.float32)
        self.flux_avg = np.zeros((self.n_nodes,2), dtype=np.float32)
        self.residual = np.zeros_like(self.d)
        self.ku = np.zeros_like(self.d)

        # create cell data
        self.grid.mesh.cell_data['alpha'] = material[:, :, 0].reshape(-1)

        # generate a finite element problem
        self.DirichBC(dirich_val)
        self.UpdateSource(f_val)
        self.NeumannBC(neumann_val)
        self.A, self.A_F, self.A_EF = self.CreateA()

    def CreateA(self):
        '''
        Stiffness matrix, return A_F and A_EF
        Subscript E means essential boundary nodes, F means all remaining valid nodes
        ''' 
        A = np.zeros((self.n_nodes,self.n_nodes))
        for i, c in enumerate(self.grid.cells):
            xe = self.grid.points[c,:].T[:2,:] #[2x4]
            alpha = self.grid.mesh.cell_data['alpha'][i]
            D = alpha*np.eye(2)
            Ke = np.zeros((4,4))
            for q in self.qpts.T:
                [_,dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp) #[2x2]
                dNdx = np.dot(dNdp, np.linalg.inv(J)) #[4x2]
                B = np.zeros((2,4))
                B[0,:] = dNdx[:,0]
                B[1,:] = dNdx[:,1]
                Ke += np.linalg.det(J)*np.dot(B.T,np.dot(D,B))
            A[np.ix_(c,c)] += Ke
            
        return A, A[np.ix_(self.grid.ndirich_valid_node,self.grid.ndirich_valid_node)], A[np.ix_(self.grid.dirich_plus_nonvalid_node,self.grid.ndirich_valid_node)]

    def ComputeAverageHeatFlux(self):
        # Initialize arrays to store total heat flux contributions and count of elements contributing
        total_flux = np.zeros((self.n_nodes, 2))
        count_contributions = np.zeros(self.n_nodes)

        qpts_flux = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        for i, c in enumerate(self.grid.cells):
            xe = self.grid.points[c, :].T[:2]  # [2, 4]
            alpha = self.grid.mesh.cell_data['alpha'][i]
            de = self.d[c].reshape(1, -1)  # [1, 4]
            for i, q in enumerate(qpts_flux.T):
                [N, dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp)  # [2, 2]
                dNdx = np.dot(dNdp, np.linalg.inv(J))  # [4, 2]
                qh = -alpha * np.dot(de, dNdx)  # [1, 2]

                # Calculate contribution at each node
                total_flux[c[i], :] += qh.squeeze()
                count_contributions[c[i]] += 1

        # Compute the average heat flux
        for node_idx in range(self.n_nodes):
            if count_contributions[node_idx] > 0:
                self.flux_avg[node_idx, :] = total_flux[node_idx, :] / count_contributions[node_idx]

        return self.flux_avg

    def ComputeHeatFlux(self):
        H = np.zeros((self.n_nodes,self.n_nodes))
        y = np.zeros((self.n_nodes,2))
        for i, c in enumerate(self.grid.cells):
            xe = self.grid.points[c,:].T[:2,:] #[2,4]
            alpha = self.grid.mesh.cell_data['alpha'][i]
            de = self.d[c].reshape(1,-1) #[1,4]
            He = np.zeros((4,4))
            for q in self.qpts.T:
                [N,dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp) #[2,2]
                dNdx = np.dot(dNdp, np.linalg.inv(J)) #[4,2]
                qh = -alpha*np.dot(de, dNdx)*np.linalg.det(J) #[1,2]
                He += np.linalg.det(J)*np.dot(N,N.T) #[4,4]
                y[c,:] += np.dot(N,qh) 
            H[np.ix_(c,c)] += He
        # set diagonal values of non-valid nodes to be 1.0  
        H[self.grid.nonvalid_node,self.grid.nonvalid_node] = 1.0
        # compute the heat flux
        self.flux[:,0] = np.linalg.solve(H, y[:,0])
        self.flux[:,1] = np.linalg.solve(H, y[:,1])
        return H

    def UpdateSource(self, f_val = None):
        '''
        Return the rhs internal sourcing term with modification from finite element term
        '''
        for c in self.grid.cells:
            xe = self.grid.points[c,:].T[:2,:] #[2x4]
            fe = np.zeros(4)
            for q in self.qpts.T:
                [N,dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp) #[2x2]
                fe += np.linalg.det(J) * N.squeeze() * f_val[c]
            self.f[c] += fe
    
    def NeumannBC(self, bc_val):
        '''
        In Neumann bc, the heat flux is assumed to flow out of the domain
        '''
        for neumann_conn in self.grid.neumann_conn_list:
            for c in neumann_conn:
                xe = self.grid.points[c,:][:,:2] #[2x2]
                le = np.linalg.norm(xe[1,:]-xe[0,:])
                for q in [1./np.sqrt(3), -1./np.sqrt(3)]:
                    N = 0.5*np.array([1-q, 1+q])
                    self.f[c] += N.squeeze() * bc_val[c] * le/2 #[2x1]

    def DirichBC(self, bc_val):
        self.d[self.grid.nonvalid_node] = 0.0 # response should be zero at nonvalid nodes
        self.d[self.grid.dirich_node] = bc_val[self.grid.dirich_node]

    def GaussianRF(self, n, a_interval):
        '''
        Only works for a square mesh
        '''
        alpha = random.uniform(2,5)
        a0, a1 = a_interval[0],a_interval[1]
        field = gr.gaussian_random_field(alpha=alpha, size=n, flag_normalize=False)
        f_min, f_max = np.min(field), np.max(field)
        rf = (a1-a0)*(field-f_min)/(f_max-f_min)+a0
        return rf

    def Solve(self, mode='direct'):
        '''
        Solve the linear equation system
        '''
        b_F = self.f[self.grid.ndirich_valid_node] - np.dot(self.A_EF.T, self.d[self.grid.dirich_plus_nonvalid_node])
        if (mode == 'direct'):
            d_F = np.linalg.solve(self.A_F, b_F)
            res_F, ku_F = np.zeros_like(d_F), np.zeros_like(d_F)
        elif (mode == 'jac'):
            d_F, res_F, ku_F = self.Jacobi(self.A_F, b_F, np.ones_like(b_F))
        self.d[self.grid.ndirich_valid_node] = d_F
        self.residual[self.grid.ndirich_valid_node] = res_F
        self.ku[self.grid.ndirich_valid_node] = ku_F

    def Jacobi(self, A, b, u0):
        Dinv = np.diag(1./np.diag(A))
        N = 0
        u = u0
        omega = 2./3.
        res = 1
        uprev = u
        ku = np.dot(A, uprev)
        residual = b - np.dot(A, uprev)
        res = np.linalg.norm(residual)
        while(res > 1e-6 and N < 1 ):
            u = omega*np.dot(Dinv, residual) + uprev
            ku = np.dot(A, u)
            residual = b - np.dot(A, u)
            res = np.linalg.norm(residual)
            self.res_arr.append(res)
            uprev = u
            print(N, res)
            N += 1

        return u, residual, ku
    
    def PlotField(self, field = None):
        '''Default is to plot the solution field'''
        if(field is None):
            field = self.d
        
        field2d = field.reshape((self.grid.img_h+1, self.grid.img_w+1))
        im = plt.imshow(field2d, origin='lower')
        
        plt.axis('off')
        plt.colorbar(im)
        plt.show()