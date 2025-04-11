import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class load_SLC(Dataset):
    def __init__(self, is_shuffle=True, is_normalize=True):
        data = np.load('SLC.npy', allow_pickle=True)

        if is_shuffle:
            np.random.shuffle(data)
        
        if is_normalize:
            data = (data - data.mean()) / ((np.sqrt(data.var()))+1e-10)
            
        self.train_x = data[:,:1000]
        self.train_x = np.expand_dims(self.train_x , axis=2)
        self.train_x = np.swapaxes(self.train_x, 1, 3)

        
    def __len__(self):
        return len(self.train_x)
        

    def __getitem__(self, idx):
        return self.train_x[idx], 1

if __name__ == "__main__":
    dataset = load_SLC()