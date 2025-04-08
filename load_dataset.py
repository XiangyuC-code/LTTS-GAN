import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os
os.chdir('/media/lscsc/nas/xiangyu/Compare/autoformer_g_2b_SE')

class my_dataset(Dataset):
    def __init__(self, split=0.8, is_norm=True, is_shuffle=True):
        awake_data = np.load('awake_data.npy')

        awake_label = np.asarray([1] * len(awake_data))

        all_data = awake_data
        all_label = awake_label

        # shuffle
        idx = np.arange(len(all_label))
        np.random.shuffle(idx)
        all_data = all_data[idx]
        all_label = all_label[idx]

        self.train_x = all_data
        self.train_y = all_label
     
        if is_norm:
            self.train_x = self.norm(self.train_x)
           
        if is_shuffle:
            self.train_x, self.train_y = self._shuffle(self.train_x, self.train_y)
        
        self.train_x = np.expand_dims(self.train_x, axis=2)
        self.train_x = np.swapaxes(self.train_x, 1, 3)


    def _shuffle(self, data, label):
        idx = np.arange(len(label))
        np.random.shuffle(idx)
        data = data[idx]
        label = label[idx]
        return data, label

    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        #B, S = epoch.shape
        #m = np.repeat(epoch.mean(axis=1), S).reshape(B,S)
        #v = np.repeat(epoch.var(axis=1), S).reshape(B,S)
        #result = (epoch - m) / ((np.sqrt(v))+e)
        result = (epoch - epoch.mean()) / ((np.sqrt(epoch.var()))+e)
        return result

    def norm(self, data):
        result = np.ones(data.shape)
        for i in range(data.shape[-1]):
            result[:,:,i] = self._normalize(data[:,:,i])
        return result
    
    def __len__(self):
        return len(self.train_y)
        

    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]
     

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