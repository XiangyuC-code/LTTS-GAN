from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
# import models_search
# import datasets
from dataLoader import *
from GANModels import Discriminator
from autoformer import Generator 
from functions import train, train_d, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger
from layers.Autoformer_EncDec import series_decomp, series_decomp_input
from scipy.spatial import distance
# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception
import argparse
from Cinc_ECG_loader import load_Cinc_ECG
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
from load_dataset import load_SLC
import numpy as np
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random 
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
    
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
def load_args():
    args = cfg.parse_args()

    args.syn_len = 128
    args.gen_bs = 16 
    args.dis_bs = 16 
    args.dataset = 'SLC' 
    args.bottom_width = 8 
    args.max_iter = 50000 * 4
    args.img_size = 32 
    args.gen_model = 'my_gen' 
    args.dis_model = 'my_dis' 
    args.df_dim = 384 
    args.d_heads = 4 
    args.d_depth = 3 
    args.g_depth = '5,4,2' 
    args.dropout = 0 
    args.latent_dim = 30
    args.gf_dim = 1024 
    args.num_workers = 16 
    args.g_lr = 0.0001 
    args.d_lr = 0.0002
    args.optimizer = 'adam' 
    args.loss = 'lsgan' 
    args.wd = 1e-3 
    args.beta1 = 0.9 
    args.beta2 = 0.999 
    args.phi = 1 
    args.batch_size = 16 
    args.num_eval_imgs = 50000 
    args.init_type = 'xavier_uniform'
    args.n_critic = 1
    args.val_freq = 20
    args.print_freq = 50 
    args.grow_steps = [0,0] 
    args.fade_in = 0 
    args.patch_size = 2 
    args.ema_kimg = 500 
    args.ema_warmup = 0.1 
    args.ema = 0.9999 
    args.diff_aug = 'translation,cutout,color' 
    args.class_name = 'SLC'
    args.exp_name = 'SLC'

    return args

def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model',default=10, help='dimension of model')
    parser.add_argument('--dropout',default=0.05)
    parser.add_argument('--factor',default=3, help='attension factor')
    parser.add_argument('--output_attention',default=False, help='whether to output attention')
    parser.add_argument('--n_heads',default=4, help='number of heads')
    parser.add_argument('--d_ff',default=40, help='dimension of fcn')
    parser.add_argument('--activation',default='gelu', help='activation')
    parser.add_argument('--d_layers',default=1, help='number of layers')
    parser.add_argument('--c_out',default=1, help='output size')
    parser.add_argument('--latent_dim',default=30, help='dimension of noise')
    parser.add_argument('--seq_len',default=128, help='sequence length')
    parser.add_argument('--moving_avg',default=[5, 25],help='window size of moving average')
    configs = parser.parse_args()
    return configs

def main():
    args = load_args()
    
#     _init_inception()
#     inception_path = check_or_download_inception(None)
#     create_inception_graph(inception_path)
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.gpu = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    configs = load_configs()
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    
    # import network
    gen_net = Generator(configs).to(args.gpu)
    print(gen_net)
    dis_net = Discriminator().to(args.gpu)
    print(dis_net)
    
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.g_lr, weight_decay=args.wd)
        
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic


    train_set = load_SLC(is_shuffle=True, is_normalize=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
    
    
    # extract features of real data
    real_data = []
    f_real = []
    for i in range(len(train_set)):
        real_data_i = train_set[i][0].squeeze(1)
        real_data_i = np.swapaxes(real_data_i,0,1)
        real_data.append(real_data_i)
    real_data = np.array(real_data, dtype=np.float32)
    
    decomp = series_decomp_input(configs.moving_avg)
    _, trend1, trend2 = decomp(torch.from_numpy(real_data))
    trend1 = trend1.to(args.gpu)
    trend1 = torch.mean(trend1, axis=0)
    trend2 = trend2.to(args.gpu)
    trend2 = torch.mean(trend2, axis=0)

    print(len(train_loader))
    
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.from_numpy(np.random.normal(0, 1, (100, args.latent_dim))).to(args.gpu).float()
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
#         avg_gen_net = deepcopy(gen_net)
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
#         del avg_gen_net
#         gen_avg_param = list(p.cuda().to(f"cuda:{args.gpu}") for p in gen_avg_param)
        
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
    # create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])
    
    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    best_epoch = 0
    for epoch in range(int(start_epoch), int(args.max_epoch)):
#         train_sampler.set_epoch(epoch)
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank==0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank==0 else 0
        
        print(str(epoch)+'|'+str(int(args.max_epoch))+'\t')
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers, trend1=trend1, trend2=trend2)
        
        # compute js distance
        gen_net.eval()
        syn_data = np.zeros([1000, args.syn_len, c])
        for gen_i in range(10):
            gen_z = torch.from_numpy(np.random.normal(0, 1, (100, args.latent_dim))).to(args.gpu).float()
            gen_imgs = gen_net(gen_z,trend1,trend2).cpu()
            gen_imgs = gen_imgs.detach().numpy()
            syn_data_i = gen_imgs.squeeze(2)
            syn_data_i = np.swapaxes(syn_data_i,1,2)

            syn_data[gen_i*100: (gen_i+1)*100] = syn_data_i

        #if epoch % 5 == 0:
        #    plotting(args,gen_net,epoch,trend1,trend2)

        #if epoch % 10 == 0:
        #    visualization(real_data[:1000],syn_data[:1000],'tsne','TSNE/',epoch)

        save_checkpoint({
            'best_epoch': best_epoch,
            'gen_model': gen_net,
            'dis_model': dis_net,
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict()
        }, 'models', filename="checkpoint_"+str(epoch)+".pth")
        

if __name__ == '__main__':
    main()
