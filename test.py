import numpy as np
from metrics import feature_extract, cos_similarity, visualization, KDE, discriminative_score
from dataLoader import unimib_load_dataset
from Cinc_ECG_loader import load_Cinc_ECG
import torch
from autoformer import Generator
from layers.Autoformer_EncDec import series_decomp
import os 
from scipy.spatial import distance
os.chdir('/media/lscsc/nas/xiangyu/Compare/autoformer_g')

gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
# load real data
real_data = []
#train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = 'Running', augment_times=None)
train_set = load_Cinc_ECG(syn_len=900)
for i in range(len(train_set)):
    real_data.append(train_set[i][0])

real_data = np.array(real_data).squeeze(2)
real_data = np.swapaxes(real_data,1,2)

decomp = series_decomp(25)
_, trend = decomp(torch.from_numpy(real_data))
trend = trend.to('cuda:0')
trend = torch.mean(trend,axis=0)

#syn_data = np.load('saved/syn_data.npy',allow_pickle=True)
f_real = []
for i in range(1):
    f_real_i = feature_extract(real_data[:,:,i])
    f_real.append(f_real_i)
    
f_real = np.array(f_real, dtype=np.float32)
f_real = np.mean(f_real, axis=1)

best_js_dist = 10
# load synthetic data
for file in os.listdir('models/'):
    if file.split('_')[-1] == 'best.pth':
        continue
    epoch = int(file.split('_')[-1].split('.')[0])
    if epoch < 55 or epoch > 2220:
        continue
    model = torch.load('models/'+file)
    gen_net = model['gen_model'].to(gpu)

    gen_net.eval()
    syn_data = np.zeros([1000, 900, 1])
    for gen_i in range(10):
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, 300))).cuda(gpu, non_blocking=True)
        gen_imgs = gen_net(gen_z,trend).cpu()
        gen_imgs = gen_imgs.detach().numpy()
        syn_data_i = gen_imgs.squeeze(2)
        syn_data_i = np.swapaxes(syn_data_i,1,2)

        syn_data[gen_i*100: (gen_i+1)*100] = syn_data_i
    
    f_syn = []
    for i in range(1):
        f_syn_i = feature_extract(syn_data[:,:,i])
        f_syn.append(f_syn_i)
    f_syn = np.array(f_syn, dtype=np.float32)

    # compute js distance
    js_dist = 0
    for i in range(1):
        for j in range(1000):
            js_dist += distance.jensenshannon(np.abs(f_real[i]), np.abs(f_syn[i,j]))
        
    js_dist /= 1*1000

    if js_dist < best_js_dist and epoch > 10:
        best_js_dist = js_dist
        best_epoch = epoch
        is_best = True
        np.save('saved/best_data.npy',syn_data)
        print('best epoch:',str(epoch),'best js distance:', str(best_js_dist))

