import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import numpy as np
from torch.utils import data
from autoformer import Generator
from Cinc_ECG_loader import load_Cinc_ECG
from layers.Autoformer_EncDec import series_decomp, series_decomp_input
import os
os.chdir("/media/lscsc/nas/xiangyu/Compare/autoformer_g_2b_SE/")

# configs - CincECG
def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model',default=10, help='dimension of model')
    parser.add_argument('--dropout',default=0.05)
    parser.add_argument('--factor',default=3, help='attension factor')
    parser.add_argument('--output_attention',default=False, help='whether to output attention')
    parser.add_argument('--n_heads',default=4, help='number of heads')
    parser.add_argument('--d_ff',default=40, help='dimension of fcn')
    parser.add_argument('--activation',default='gelu', help='activation')
    parser.add_argument('--d_layers',default=3, help='number of layers')
    parser.add_argument('--c_out',default=1, help='output size')
    parser.add_argument('--latent_dim',default=300, help='dimension of noise')
    parser.add_argument('--seq_len',default=900, help='sequence length')
    parser.add_argument('--moving_avg',default=[25, 125],help='window size of moving average')
    configs = parser.parse_args()
    return configs

configs = load_configs()

# load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torch.load("saved_25,125/checkpoint_81.pth")
gen_net = model['gen_model'].to(device)

print(gen_net)

# load dataset

train_set = load_Cinc_ECG(syn_len=configs.seq_len)
train_loader = data.DataLoader(train_set, batch_size=32, num_workers=16, shuffle = True)

real_data = []
f_real = []
for i in range(len(train_set)):
    real_data_i = train_set[i][0].squeeze(1)
    real_data_i = np.swapaxes(real_data_i,0,1)
    real_data.append(real_data_i)
real_data = np.array(real_data, dtype=np.float32)
    
decomp = series_decomp_input(configs.moving_avg)
_, trend1, trend2 = decomp(torch.from_numpy(real_data))
trend1 = trend1.to(device)
trend1 = torch.mean(trend1, axis=0)
trend2 = trend2.to(device)
trend2 = torch.mean(trend2, axis=0)

# visulization
gen_net.eval()
z = torch.from_numpy(np.random.normal(0, 1, (32, configs.latent_dim))).to(device).float()
gen_img = gen_net(z, trend1, trend2)

print("done")