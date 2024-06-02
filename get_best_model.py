import numpy as np
from dataLoader import unimib_load_dataset
import torch
from autoformer import Generator
from metrics import feature_extract
from scipy.spatial import distance
from layers.Autoformer_EncDec import series_decomp
import os
os.chdir('/media/lscsc/nas/xiangyu/Compare/autoformer_g')

# load real data
real_data = []
train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = 'Running', augment_times=None)
for i in range(len(train_set)):
    real_data.append(train_set[i][0])

real_data = np.array(real_data[:1000]).squeeze(2)
real_data = np.swapaxes(real_data,1,2)

decomp = series_decomp(25)
_, trend = decomp(torch.from_numpy(real_data))
trend = trend.to("cuda:0")
trend = torch.mean(trend,axis=0)


# extract features
n, seq_len, c = real_data.shape
f_real = []
for i in range(3):
    f_real_i = feature_extract(real_data[:,:,i])
    f_real.append(f_real_i)

f_real = np.array(f_real, dtype=np.float32)
f_real = np.mean(f_real, axis=1)

best_js_dis = 10
best_epoch = 0

for epoch in range(300):
    # load synthetic data
    model = torch.load('models/checkpoint_'+str(epoch)+'.pth')
    gen_net = model['gen_model']

    gen_net.eval()
    gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (1000, 100))).cuda("cuda:0", non_blocking=True)
    gen_imgs = gen_net(gen_z,trend).cpu()
    gen_imgs = gen_imgs.detach().numpy()
    syn_data = gen_imgs.squeeze(2)
    syn_data = np.swapaxes(syn_data,1,2)

    # extract features
    n, seq_len, c = real_data.shape
    f_syn = []

    for i in range(3):
        f_syn_i = feature_extract(syn_data[:,:,i])
        f_syn.append(f_syn_i)

    f_syn = np.array(f_syn, dtype=np.float32)

    # compute js distance
    js_dist = 0
    for i in range(c):
        for j in range(1000):
            js_dist += distance.jensenshannon(np.abs(f_real[i]), np.abs(f_syn[i,j]))

    js_dist /= c*1000

    if js_dist < best_js_dis:
        best_js_dis = js_dist
        best_epoch = epoch
        print('best_js_dis:',best_js_dis,'\tbest_epoch:',best_epoch)

print('Best_js_dis:',best_js_dis,'\tBest_epoch:',best_epoch)

