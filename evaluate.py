import numpy as np 
import torch
from torch.utils import data
from layers.Autoformer_EncDec import series_decomp, series_decomp_input
from metrics import feature_extract, cos_similarity, visualization, mmd_rbf
from dataLoader import unimib_load_dataset
from Cinc_ECG_loader import load_Cinc_ECG
from load_dataset import my_dataset, load_SLC
from scipy.spatial import distance
import os 

os.chdir('/media/lscsc/nas/xiangyu/Compare/autoformer_g_2b_SE')
gpu = "cuda:1" if torch.cuda.is_available() else "cpu"
# load real data
#train_set = my_dataset()
#train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = 'Running', augment_times=None)
# train_set = load_Cinc_ECG(syn_len = 1500)
train_set = load_SLC(is_shuffle=True, is_normalize=True)


ori_data = []
for i in range(len(train_set)):
        data_i = train_set[i][0].squeeze(1)
        data_i = np.swapaxes(data_i,0,1)
        ori_data.append(data_i)
ori_data = np.array(ori_data, dtype=np.float32)

# extract feature
f_real = []
for i in range(1):
    f_real_i = feature_extract(ori_data[:,:,i])
    f_real.append(f_real_i)
f_real = np.array(f_real, dtype=np.float32)
f_real = np.mean(f_real, axis=1)

decomp = series_decomp_input([5,25])
_, trend1, trend2 = decomp(torch.from_numpy(ori_data))
trend1 = trend1.to(gpu)
trend1 = torch.mean(trend1, axis=0)
trend2 = trend2.to(gpu)
trend2 = torch.mean(trend2, axis=0)

'''
# load model
model = torch.load('models/checkpoint_426.pth')
gen_net = model['gen_model'].to(gpu)



gen_net.eval()
gen_z = torch.from_numpy(np.random.normal(0, 1, (1000, 300))).cuda(gpu).float()
gen_imgs = gen_net(gen_z,trend1,trend2).cpu()
gen_imgs = gen_imgs.detach().numpy()
syn_data = gen_imgs.squeeze(2)
syn_data = np.swapaxes(syn_data, 1, 2)

visualization(ori_data[:1000],syn_data[:1000],'tsne','./',538)
'''
# generate synthetic data
best_metrics = 10

for file in os.listdir('models/'):
    if file.split('_')[-1] == 'best.pth':
        continue
    
    epoch = int(file.split('_')[-1].split('.')[0])

    if epoch <50 or epoch > 3000:
        continue
    
    model = torch.load('models/'+file)
    gen_net = model['gen_model'].to(gpu)
   
    gen_net.eval()
    gen_z = torch.from_numpy(np.random.normal(0, 1, (1000, 300))).cuda(gpu).float()
    gen_imgs = gen_net(gen_z,trend1,trend2).cpu()
    gen_imgs = gen_imgs.detach().numpy()
    syn_data = gen_imgs.squeeze(2)
    syn_data = np.swapaxes(syn_data,1,2)

    # compute mmd
    mmd = []
    real_data = torch.tensor(ori_data[:1000])
    syn_data = torch.tensor(syn_data[:1000])
    for i in range(1):
        mmd.append(mmd_rbf(real_data[:,:,i],syn_data[:,:,i]))
    mmd = np.mean(mmd)     
    

    # extract features
    f_syn = []
    for i in range(1):
        f_syn_i = feature_extract(syn_data[:,:,i])
        f_syn.append(f_syn_i)
    f_syn = np.array(f_syn, dtype=np.float32)

    # compute js distance
    c=1
    js_dist = 0
    for i in range(c):
        for j in range(1000):
            js_dist += distance.jensenshannon(np.abs(f_real[i]), np.abs(f_syn[i,j]))

    js_dist /= c*1000

    metrics = (js_dist + mmd)/2
    if metrics < best_metrics and epoch > 10:
        best_metrics = metrics
        best_epoch = epoch
        is_best = True
        np.save('saved/best_data.npy',syn_data)
        print('best epoch:',str(epoch),'best metrics:', str(best_metrics)) 
        print('mmd:',str(mmd),'best JSD:', str(js_dist),'\n') 


# visulization metrics
#visualization(ori_data[:1000], syn_data, 'tsne', 'T-sne_', 0)

# save syn data
#np.save('syn_data.npy',syn_data)

'''
# load parameters
parameters = model.load_parameters('Running_21')

# build model
Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim)

discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 
                  'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings, 
        kappa, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size, 
        total_examples=1325, l2norm_bound=l2norm_bound,
        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG, parameters=parameters)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# sample
vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
'''