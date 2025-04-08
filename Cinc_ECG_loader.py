import numpy as np
import torch
import torch.utils.data as data
import os
os.chdir('/media/lscsc/nas/xiangyu/Compare/tts-gan-main')

class load_Cinc_ECG(data.Dataset):
    def __init__(self, is_shuffle=True, is_normalize=True, syn_len=None) -> None:
        super().__init__()

        train = np.loadtxt('Cinc_ECG/train.txt', dtype=np.float32)
        if not syn_len:
            syn_len = train.shape[-1]

        train_label = train[:,0]
        train_data = train[:,1:syn_len + 1]

        test = np.loadtxt('Cinc_ECG/test.txt', dtype=np.float32)
        test_label = test[:,0]
        test_data = test[:,1:syn_len + 1]

        all_data = np.r_[train_data, test_data]
        all_label = np.r_[train_label, test_label]

        if is_shuffle:
            idx = np.arange(len(all_label))
            np.random.shuffle(idx)
            all_data = all_data[idx]
            all_label = all_label[idx]

        if is_normalize:
            all_data = self._normalize(all_data)
        
        B, S = all_data.shape
        self.data = all_data.reshape(B, 1, 1, S)
        self.label = all_label

    def _normalize(self, data):
        e = 1e-10
        data = (data - data.mean()) / np.sqrt((data.var() - e))
        return data

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

if __name__ == '__main__':
    dataset = load_Cinc_ECG()