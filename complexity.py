import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format
from autoformer import Generator 
import cfg
import argparse


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
    parser.add_argument('--c_out',default=3, help='output size')
    parser.add_argument('--latent_dim',default=300, help='dimension of noise')
    parser.add_argument('--seq_len',default=1500, help='sequence length')
    parser.add_argument('--moving_avg',default=[5, 25],help='window size of moving average')
    configs = parser.parse_args()
    return configs

configs = load_configs()
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Generator(configs).to(device)
input_shape = (1000)
# summary(net, input_shape)

input_tensor = torch.randn(1, configs.latent_dim).to(device)
trend1 = torch.randn(configs.seq_len, configs.c_out).to(device)
trend2 = torch.randn(configs.seq_len, configs.c_out).to(device)

flops, params = profile(net, inputs=(input_tensor, trend1, trend2))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" %(flops))
print("params: %s" %(params))
