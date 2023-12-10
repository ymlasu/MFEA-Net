import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MechanicalDataSet(Dataset):
    def __init__(self, h5file, device, transform=None, target_transform=None):
        h5 = h5py.File(h5file, 'r')
        self.device = device
        self.mask = np.array(h5['mask'], dtype=np.double)
        self.dirich_idx = np.array(h5['dirich_idx'], dtype=np.double)
        self.dirich_value = np.array(h5['dirich_value'], dtype=np.double)
        self.traction_value = np.array(h5['neumann_value'], dtype=np.double)
        self.traction_conn = np.array(h5['neumann_conn'], dtype=np.int64) 
        self.material = np.array(h5['material'], dtype=np.double)
        self.source = np.array(h5['source'], dtype=np.double)
        self.solution = np.array(h5['solution'], dtype=np.double)
        self.totensor = ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.source.shape[0]

    def __getitem__(self, idx):
        source_tensor = torch.flip(self.totensor(self.source[idx]), dims=[1]).to(self.device)
        solution_tensor = torch.flip(self.totensor(self.solution[idx]), dims=[1]).to(self.device)
        material_tensor = torch.flip(self.totensor(self.material[idx]), dims=[1]).to(self.device)
        dirich_value_tensor = torch.flip(self.totensor(self.dirich_value[idx]), dims=[1]).to(self.device)
        dirich_idx_tensor = torch.flip(self.totensor(self.dirich_idx[idx]), dims=[1]).to(self.device)
        traction_value_tensor = torch.flip(self.totensor(self.traction_value[idx]), dims=[1]).to(self.device)
        traction_conn_tensor = self.totensor(self.traction_conn[idx]).to(self.device)
        mask_tensor = torch.flip(self.totensor(self.mask[idx]), dims=[1]).to(self.device)
        if self.transform:
            source_tensor = self.transform(source_tensor)
            solution_tensor = self.transform(solution_tensor)
            material_tensor = self.transform(material_tensor)
            traction_value_tensor = self.transform(traction_value_tensor)
            dirich_value_tensor = self.transform(dirich_value_tensor)
            dirich_idx_tensor = self.transform(dirich_idx_tensor)
            mask_tensor = self.transform(mask_tensor)
        return mask_tensor, dirich_idx_tensor, dirich_value_tensor, traction_value_tensor, traction_conn_tensor, material_tensor, source_tensor, solution_tensor
    