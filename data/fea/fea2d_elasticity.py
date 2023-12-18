import numpy as np
import random
import matplotlib.pyplot as plt
import fea.gaussian_random_fields as gr

class ElasticFEM2D():
    '''
    Note: input the mesh grid, material, bc and f values, generate a finite element problem
    '''
    def __init__(self, grid, mode, material, dirich_val, neumann_val, f_val):
        self.grid = grid
        self.mode = mode # plane stress or plane strain
        self.res_arr = []
        self.qpts = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])/np.sqrt(3) #[2x4], integration points
        self.n_nodes = grid.points.shape[0]
        self.d = np.zeros(2*self.n_nodes, dtype=np.float32) # x,y components
        self.f = np.zeros_like(f_val, dtype=np.float32)
        self.strain = np.zeros((self.n_nodes,3), dtype=np.float32)
        self.stress = np.zeros((self.n_nodes,3), dtype=np.float32)
        self.strain_energy_density = np.zeros((self.n_nodes,), dtype=np.float32)
        self.residual = np.zeros_like(self.d)
        self.ku = np.zeros_like(self.d)

        # create cell data
        self.grid.mesh.cell_data['E'] = material[:, :, 0].reshape(-1)
        self.grid.mesh.cell_data['v'] = material[:, :, 1].reshape(-1)

        # generate a finite element problem
        self.DirichBC(dirich_val)
        self.UpdateBodyForce(f_val)
        self.NeumannBC(neumann_val)
        self.A, self.A_F, self.A_EF = self.CreateA()

    def StiffnessMatrix(self, E, v):
        if(self.mode == 'plane_strain'):
            D = E/(1.+v)/(1-2*v)*np.array([[1.-v, v, 0.],
                                           [v, 1.-v, 0.],
                                           [0., 0., (1-2*v)/2]])
        elif(self.mode == 'plane_stress'):
            D = E/(1.-v*v)*np.array([[1., v, 0.], 
                                     [v, 1., 0.], 
                                     [0., 0., (1.-v)/2.]]) 
        return D
    
    def CreateA(self):
        '''
        Stiffness matrix, return A_F and A_EF
        Subscript E means essential boundary nodes, F means all remaining valid nodes
        ''' 
        A = np.zeros((2*self.n_nodes,2*self.n_nodes))
        for i, c in enumerate(self.grid.cells):
            xe = self.grid.points[c,:].T[:2,:] #[2x4]
            E = self.grid.mesh.cell_data['E'][i]
            v = self.grid.mesh.cell_data['v'][i]
            D = self.StiffnessMatrix(E, v)
            Ke = np.zeros((8,8))
            for q in self.qpts.T:
                [_,dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp) #[2x2]
                dNdx = np.dot(dNdp, np.linalg.inv(J)) #[4x2]
                B = np.zeros((3,8))
                B[0, 0::2] = dNdx[:,0]
                B[1, 1::2] = dNdx[:,1]
                B[2, 0::2] = dNdx[:,1]
                B[2, 1::2] = dNdx[:,0]
                Ke += np.linalg.det(J)*np.dot(B.T,np.dot(D,B))
            cc = np.zeros((8,), dtype=int)
            cc[0::2] = 2*c
            cc[1::2] = 2*c+1
            A[np.ix_(cc,cc)] += Ke
            
        return A, A[np.ix_(self.grid.df_ndirich_valid,self.grid.df_ndirich_valid)], A[np.ix_(self.grid.df_dirich_plus_nonvalid,self.grid.df_ndirich_valid)]
        
        
    def UpdateBodyForce(self, f_val = None):
        '''
        Input: f_val is a 1-D array that contains both x and y degree of freedom
        '''
        for c in self.grid.cells:
            xe = self.grid.points[c,:].T[:2,:] #[2x4]
            for q in self.qpts.T:
                [N,dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp) #[2x2]
                self.f[2*c] += np.linalg.det(J)*np.dot(N,np.dot(N.T,f_val[2*c]))
                self.f[2*c+1] += np.linalg.det(J)*np.dot(N,np.dot(N.T,f_val[2*c+1]))

    def ComputeStrainStress(self):
        total_strain = np.zeros((self.n_nodes, 3))
        total_stress = np.zeros((self.n_nodes, 3))
        count_contributions = np.zeros(self.n_nodes)
        qpts_strain = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        for i, c in enumerate(self.grid.cells):
            xe = self.grid.points[c, :].T[:2]  # [2, 4]
            E = self.grid.mesh.cell_data['E'][i]
            v = self.grid.mesh.cell_data['v'][i]
            D = self.StiffnessMatrix(E, v)
            de = np.zeros((8, 1))
            de[0::2] = self.d[2*c].reshape(-1, 1) # x
            de[1::2] = self.d[2*c+1].reshape(-1, 1) # y
            for i, q in enumerate(qpts_strain.T):
                [N, dNdp] = self.grid.shapefunc(q)
                J = np.dot(xe, dNdp)  # [2, 2]
                dNdx = np.dot(dNdp, np.linalg.inv(J))  # [4, 2]
                B = np.zeros((3,8))
                B[0, 0::2] = dNdx[:,0]
                B[1, 1::2] = dNdx[:,1]
                B[2, 0::2] = dNdx[:,1]
                B[2, 1::2] = dNdx[:,0]
                strain_h = np.dot(B, de)  # [3, 1]
                stress_h = np.dot(D, strain_h) # [3, 1]

                # Calculate contribution at each node
                total_strain[c[i], :] += strain_h.squeeze()
                total_stress[c[i], :] += stress_h.squeeze()
                count_contributions[c[i]] += 1

        # Compute the average strain, and stress
        for node_idx in range(self.n_nodes):
            if count_contributions[node_idx] > 0:
                self.strain[node_idx, :] = total_strain[node_idx, :] / count_contributions[node_idx]
                self.stress[node_idx, :] = total_stress[node_idx, :] / count_contributions[node_idx]
    
    def ComputeStrainEnergyDensity(self):
        self.strain_energy_density = 0.5*np.sum(self.strain * self.stress, axis=1)

    def NeumannBC(self, bc_val):
        '''
        Input: bc_val is a 1-D array that contains both x and y degree of freedom
        '''
        for neumann_conn in self.grid.neumann_conn_list:
            for c in neumann_conn:
                xe = self.grid.points[c,:][:,:2] #[2x2]
                le = np.linalg.norm(xe[1,:]-xe[0,:])
                for q in [1./np.sqrt(3), -1./np.sqrt(3)]:
                    N = 0.5*np.array([1-q, 1+q])
                    self.f[2*c] += np.dot(N,np.dot(N.T,bc_val[2*c]))*le/2 
                    self.f[2*c+1] += np.dot(N,np.dot(N.T,bc_val[2*c+1]))*le/2 

    def DirichBC(self, bc_val):
        self.d[self.grid.df_nonvalid] = 0.0 # response should be zero at nonvalid nodes
        self.d[self.grid.df_dirich] = bc_val[self.grid.df_dirich]

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
        b_F = self.f[self.grid.df_ndirich_valid] - np.dot(self.A_EF.T, self.d[self.grid.df_dirich_plus_nonvalid])
        if (mode == 'direct'):
            d_F = np.linalg.solve(self.A_F, b_F)
            res_F, ku_F = np.zeros_like(d_F), np.zeros_like(d_F)
        elif (mode == 'jac'):
            d_F, res_F, ku_F = self.Jacobi(self.A_F, b_F, np.zeros_like(b_F))
        self.d[self.grid.df_ndirich_valid] = d_F
        self.residual[self.grid.df_ndirich_valid] = res_F
        self.ku[self.grid.df_ndirich_valid] = ku_F

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
        while(res > 1e-6 and N < 1000):
            u = omega*np.dot(Dinv, residual) + uprev
            ku = np.dot(A, u)
            residual = b - np.dot(A, u)
            res = np.linalg.norm(residual)
            self.res_arr.append(res)
            uprev = u
            print(N, res)
            N += 1

        return u, residual, ku
    
    def PlotField(self, num = 1, field = None):
        '''Default is to plot the solution field'''
        if(field is None):
            field = self.d
        
        h, w = self.grid.img_h+1, self.grid.img_w+1
        fig = plt.figure()

        if(num == 1):
            field2d = field.reshape((h, w))
            im = plt.imshow(field2d, origin='lower')
            plt.colorbar(im)

        if(num == 2):
            field1 = field[0::2].reshape((h, w))
            field2 = field[1::2].reshape((h, w))

            fig.add_subplot(1,2,1)
            im1 = plt.imshow(field1, origin='lower')
            plt.title('X')
            plt.colorbar(im1)

            fig.add_subplot(1,2,2)
            im2 = plt.imshow(field2, origin='lower')
            plt.title('Y')
            plt.colorbar(im2)

        if(num == 3):
            field1 = field[:,0].reshape((h, w))
            field2 = field[:,1].reshape((h, w))
            field3 = field[:,2].reshape((h, w))

            fig.add_subplot(1,3,1)
            im1 = plt.imshow(field1, origin='lower')
            plt.title('X')
            plt.colorbar(im1)

            fig.add_subplot(1,3,2)
            im2 = plt.imshow(field2, origin='lower')
            plt.title('Y')
            plt.colorbar(im2)

            fig.add_subplot(1,3,3)
            im3 = plt.imshow(field3, origin='lower')
            plt.title('XY')
            plt.colorbar(im3)

        plt.axis('off')
        plt.tight_layout()
        plt.show()