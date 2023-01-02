import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ThermalDataSet(Dataset):
    def __init__(self, h5file, transform=None, target_transform=None):
        h5 = h5py.File(h5file, 'r')
        self.dirich_idx = np.array(h5['dirich_idx'], dtype=np.double)
        self.dirich_value = np.array(h5['dirich_value'], dtype=np.double)
        self.traction_idx = np.array(h5['neumann_idx'], dtype=np.double)
        self.traction_value = np.array(h5['neumann_value'], dtype=np.double)
        self.material = np.array(h5['material'], dtype=np.double)
        self.source = np.array(h5['source'], dtype=np.double)
        self.solution = np.array(h5['solution'], dtype=np.double)
        self.totensor = ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.source.shape[0]

    def __getitem__(self, idx):
        source_tensor = torch.flip(self.totensor(self.source[idx]), dims=[1])
        solution_tensor = torch.flip(self.totensor(self.solution[idx]), dims=[1])
        material_tensor = torch.flip(self.totensor(self.material[idx]), dims=[1])
        traction_value_tensor = torch.flip(self.totensor(self.traction_value[idx]), dims=[1])
        traction_idx_tensor = torch.flip(self.totensor(self.traction_idx[idx]), dims=[1])
        dirich_value_tensor = torch.flip(self.totensor(self.dirich_value[idx]), dims=[1])
        dirich_idx_tensor = torch.flip(self.totensor(self.dirich_idx[idx]), dims=[1])
        if self.transform:
            source_tensor = self.transform(source_tensor)
            solution_tensor = self.transform(solution_tensor)
            material_tensor = self.transform(material_tensor)
            traction_idx_tensor = self.transform(traction_idx_tensor)
            traction_value_tensor = self.transform(traction_value_tensor)
            dirich_value_tensor = self.transform(dirich_value_tensor)
            dirich_idx_tensor = self.transform(dirich_idx_tensor)
        return dirich_idx_tensor, dirich_value_tensor, traction_idx_tensor, traction_value_tensor, material_tensor, source_tensor, solution_tensor
    
class ElasticityDataSet(Dataset):
    '''
    Dataset stores dirich_idx, dirich_value, traction_idx, traction_value, material, body_force, solution
    Tensor shape will become (2, h, w) after tranformed to be tensor
    '''
    def __init__(self, h5file, transform=None, target_transform=None):
        h5 = h5py.File(h5file, 'r')
        self.dirich_idx = np.array(h5['dirich_idx'], dtype=np.double)
        self.dirich_value = np.array(h5['dirich_value'], dtype=np.double)
        self.traction_idx = np.array(h5['traction_idx'], dtype=np.double)
        self.traction_value = np.array(h5['traction_value'], dtype=np.double)
        self.material = np.array(h5['material'], dtype=np.double)
        self.body_force = np.array(h5['body_force'], dtype=np.double)
        self.solution = np.array(h5['solution'], dtype=np.double)
        self.totensor = ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.body_force.shape[0]

    def __getitem__(self, idx):
        body_force_tensor = torch.flip(self.totensor(self.body_force[idx]), dims=[1])
        solution_tensor = torch.flip(self.totensor(self.solution[idx]), dims=[1])
        material_tensor = torch.flip(self.totensor(self.material[idx]), dims=[1])
        traction_value_tensor = torch.flip(self.totensor(self.traction_value[idx]), dims=[1])
        traction_idx_tensor = torch.flip(self.totensor(self.traction_idx[idx]), dims=[1])
        dirich_value_tensor = torch.flip(self.totensor(self.dirich_value[idx]), dims=[1])
        dirich_idx_tensor = torch.flip(self.totensor(self.dirich_idx[idx]), dims=[1])
        if self.transform:
            body_force_tensor = self.transform(body_force_tensor)
            solution_tensor = self.transform(solution_tensor)
            material_tensor = self.transform(material_tensor)
            traction_idx_tensor = self.transform(traction_idx_tensor)
            traction_value_tensor = self.transform(traction_value_tensor)
            dirich_value_tensor = self.transform(dirich_value_tensor)
            dirich_idx_tensor = self.transform(dirich_idx_tensor)
        return dirich_idx_tensor, dirich_value_tensor, traction_idx_tensor, traction_value_tensor, material_tensor, body_force_tensor, solution_tensor