import numpy as np
import os
os.chdir('/media/lscsc/nas/xiangyu/Compare/tts-gan-main')


train = np.loadtxt('Cinc_ECG/train.txt', dtype=np.float32)
syn_len = 900

train_label = train[:,0]
train_data = train[:,1:syn_len + 1]

test = np.loadtxt('Cinc_ECG/test.txt', dtype=np.float32)
test_label = test[:,0]
test_data = test[:,1:syn_len + 1]

all_data = np.r_[train_data, test_data]
all_label = np.r_[train_label, test_label]


idx = np.arange(len(all_label))
np.random.shuffle(idx)
all_data = all_data[idx]
all_label = all_label[idx]

all_data = np.expand_dims(all_data,2)

l = len(all_data)
d = {'pdf': None, 'labels': {'train': None, 'test': None, 'vali': None}}
samples = {}
samples['train'] = all_data[:int(0.8 * l)]
samples['test'] = all_data[int(0.8 * l):int(0.9 * l)]
samples['vali'] = all_data[int(0.9 * l):]
d['samples'] = samples
np.save('CECG.npy', d)
