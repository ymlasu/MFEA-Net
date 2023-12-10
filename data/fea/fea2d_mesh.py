import meshio
import numpy as np

class SquareMeshThermal():
    """ 
    Handle a simple square mesh with quad elements
    Given input of image, boundary nodes, and connections
    """

    def __init__(self, hsize, mask, dirich_idx, neumann_conn_list, outfile = None):
        self.hsize = hsize
        self.img_h, self.img_w = mask.shape
        self.mask = mask
        self.dirich_idx = dirich_idx # 2D map, Dirichlet bc nodes are 1
        self.neumann_conn_list = neumann_conn_list
        self.points, self.cells = self.generate_mesh()
        self.generate_node_index()
        
        if outfile is not None:
            self.save_mesh(outfile)

    def generate_mesh(self):
        x = np.linspace(0,self.hsize * self.img_w,self.img_w+1, dtype=np.float64)
        y = np.linspace(0,self.hsize * self.img_h,self.img_h+1, dtype=np.float64)
        ms_x, ms_y = np.meshgrid(x,y)
        x = np.ravel(ms_x).reshape(-1,1)
        y = np.ravel(ms_y).reshape(-1,1)
        z = np.zeros_like(x, dtype=np.float64)
        points = np.concatenate((x,y,z),axis=1)
        n_element = self.img_h * self.img_w
        nodes = np.linspace(0,points.shape[0],points.shape[0],endpoint=False,dtype=int).reshape(self.img_h+1,self.img_w+1)
        cells = np.zeros((n_element,4),dtype=int)
        cells[:,0] = np.ravel(nodes[:self.img_w,:self.img_h])
        cells[:,1] = np.ravel(nodes[:self.img_w,1:])
        cells[:,2] = np.ravel(nodes[1:,1:])
        cells[:,3] = np.ravel(nodes[1:,:self.img_h])
        has_element = self.mask > 1e-6 # elements that occupied by materials
        cells = cells[has_element.reshape(-1)]
        self.mesh = meshio.Mesh(points, [("quad",cells)]) # generate a mesh used for output
        return points, cells
    
    def generate_node_index(self):
        self.node_list = np.arange((self.img_h+1)*(self.img_w+1))
        self.valid_node = np.unique(self.cells.reshape(-1))
        self.nonvalid_node = np.setdiff1d(self.node_list,self.valid_node)

        node_idx = np.where(self.dirich_idx.reshape(-1) == 1) # dirich node index
        self.dirich_node = self.node_list[node_idx]
        self.dirich_plus_nonvalid_node = np.concatenate((self.dirich_node,self.nonvalid_node))
        self.ndirich_valid_node = np.setdiff1d(self.valid_node,self.dirich_node)

    def shapefunc(self, p):
        # shape function
        N = 0.25*np.array([[(1-p[0])*(1-p[1])],
                           [(1+p[0])*(1-p[1])],
                           [(1+p[0])*(1+p[1])],
                           [(1-p[0])*(1+p[1])]])

        dNdp = 0.25*np.array([[-(1-p[1]), -(1-p[0])],
                              [(1-p[1]), -(1+p[0])],
                              [(1+p[1]), (1+p[0])],
                              [-(1+p[1]), (1-p[0])]])
        return N, dNdp

    def save_mesh(self,outfile = 'mesh_square.vtk'):
        self.mesh.write(outfile)


class SquareMeshElastic():
    """ 
    Handle a simple square mesh with quad elements
    Given input of image, boundary nodes, and connections
    """

    def __init__(self, hsize, mask, dirich_idx, neumann_conn_list, outfile = None):
        '''The mask shows the element mask, indicate whether there is any valid element'''
        self.hsize = hsize
        self.img_h, self.img_w = mask.shape
        self.mask = mask # mask has the value of 0 or 1
        self.dirich_idx = dirich_idx # Dirichlet idx has shape of (h, w, 2)
        self.neumann_conn_list = neumann_conn_list # list of neumann_conn
        self.points, self.cells = self.generate_mesh()
        self.generate_node_index()
        
        if outfile is not None:
            self.save_mesh(outfile)

    def generate_mesh(self):
        x = np.linspace(0,self.hsize * self.img_w,self.img_w+1, dtype=np.float64)
        y = np.linspace(0,self.hsize * self.img_h,self.img_h+1, dtype=np.float64)
        ms_x, ms_y = np.meshgrid(x,y)
        x = np.ravel(ms_x).reshape(-1,1)
        y = np.ravel(ms_y).reshape(-1,1)
        z = np.zeros_like(x, dtype=np.float64)
        points = np.concatenate((x,y,z),axis=1)
        n_element = self.img_h * self.img_w
        nodes = np.linspace(0,points.shape[0],points.shape[0],endpoint=False,dtype=int).reshape(self.img_h+1,self.img_w+1)
        cells = np.zeros((n_element,4),dtype=int)
        cells[:,0] = np.ravel(nodes[:self.img_w,:self.img_h])
        cells[:,1] = np.ravel(nodes[:self.img_w,1:])
        cells[:,2] = np.ravel(nodes[1:,1:])
        cells[:,3] = np.ravel(nodes[1:,:self.img_h])
        has_element = self.mask > 1e-6 # elements that occupied by materials
        cells = cells[has_element.reshape(-1)]
        self.mesh = meshio.Mesh(points, [("quad",cells)]) # generate a mesh used for output
        return points, cells
    
    def generate_node_index(self):
        # generate global node index 
        self.nodes = np.arange((self.img_h+1)*(self.img_w+1))
        self.valid_node = np.unique(self.cells.reshape(-1))
        self.nonvalid_node = np.setdiff1d(self.nodes,self.valid_node)

        # generate global dof
        self.df_nodes = np.repeat(2*self.nodes,2)+np.tile([0,1],self.nodes.shape[0])
        self.df_valid = np.repeat(2*self.valid_node,2)+np.tile([0,1],self.valid_node.shape[0])
        self.df_nonvalid = np.repeat(2*self.nonvalid_node,2)+np.tile([0,1],self.nonvalid_node.shape[0])
        
        # generate boundary dof
        dof_idx = np.where(self.dirich_idx.reshape(-1) == 1) # dirich dof index
        self.df_dirich = self.df_nodes[dof_idx]
        self.df_dirich_plus_nonvalid = np.concatenate((self.df_dirich,self.df_nonvalid))
        self.df_ndirich_valid = np.setdiff1d(self.df_valid,self.df_dirich)

    def shapefunc(self, p):
        # shape function
        N = 0.25*np.array([[(1-p[0])*(1-p[1])],
                           [(1+p[0])*(1-p[1])],
                           [(1+p[0])*(1+p[1])],
                           [(1-p[0])*(1+p[1])]])

        dNdp = 0.25*np.array([[-(1-p[1]), -(1-p[0])],
                              [(1-p[1]), -(1+p[0])],
                              [(1+p[1]), (1+p[0])],
                              [-(1+p[1]), (1-p[0])]])
        return N, dNdp

    def save_mesh(self,outfile = 'mesh_square.vtk'):
        self.mesh.write(outfile)