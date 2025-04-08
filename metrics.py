import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from GANModels import Discriminator
import torch
import torch.nn as nn


import torch
 
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵,即上文中的K
    Params:
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)
 
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def feature_extract(data):
    '''
    Extract feature vector of the input signal.

    Augs:
      - data: input signal.
    
    Return:
      - feature: features.
    '''

    data = np.array(data, dtype=np.float32)

    n = len(data)
    feature = []

    for i in range(n):
        median = np.median(data[i])
        mean = np.mean(data[i])
        std = np.std(data[i])
        var = np.var(data[i])
        rms = np.sqrt(np.mean(data[i]**2))
        maximum = np.max(data[i])
        minimum = np.min(data[i])

        feature.append([median, mean, std, var, rms, maximum, minimum])
    
    return feature


def cos_similarity(f_real, f_syn):
    '''
    Compute the cosine similarity between features of real data and synthetic data.

    Augs:
      - f_real: Features of real data.
      - f_syn: Features of synthetic data.

    Returns:
      - cos_sim: Average cosine similarity score.
    '''

    f_real = np.array(f_real, dtype=np.float32)
    f_syn = np.array(f_syn, dtype=np.float32)

    n = len(f_syn)
    cos_sim = 0

    # compute cosine similarities for each pair of real signal feature and synthetic signal feature
    for i in range(n):
        cos_sim += np.sum(f_real * f_syn[i]) / np.sqrt(np.sum(f_real**2) * np.sum(f_syn[i]**2))

    cos_sim /= n

    return cos_sim
      
      
def visualization (ori_data, generated_data, analysis, path, epoch):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  

    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)#.reshape([ori_data.shape[0],ori_data.shape[-1],ori_data.shape[1]])
    generated_data = np.asarray(generated_data).reshape(ori_data.shape)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
#         plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
#         plt.show()    
        
    plt.savefig(path + f'{epoch}.png')
    plt.show()

def KDE(ori_data,generated_data1,epoch):

    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    ori_data = ori_data[:,:,:-1]
    generated_data1 = np.asarray(generated_data1)

    ori_data = ori_data[idx]
    generated_data1 = generated_data1[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data1[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (prep_data_hat, np.reshape(np.mean(generated_data1[i, :, :], 1), [1, seq_len]))
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    f, ax = plt.subplots(1)
    sns.distplot(prep_data, hist = False, kde = True,kde_kws = {'linewidth': 6},label = 'Original')
    sns.distplot(prep_data_hat, hist = False, kde = True,kde_kws = {'linewidth': 6,'linestyle':'--'},label = 'GT-GAN')
    # Plot formatting
    plt.legend(prop={'size': 22})
    plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig("saved/"+str(epoch)+"_histo.png", dpi=100,bbox_inches='tight')
    plt.close()


def train_test_devide(real, syn, rate = 0.8):
    real = np.asarray(real)
    syn = np.asarray(syn)

    len_r = len(real)
    train_real = real[:int(len_r*rate)]
    test_real = real[int(len_r*rate):]

    len_s= len(syn)
    train_syn = syn[:int(len_s*rate)]
    test_syn = syn[int(len_s*rate):]

    train_x = np.r_[train_real,train_syn]
    train_y = np.array([1]*train_real.shape[0] + [0]*train_syn.shape[0])
    test_x = np.r_[test_real,test_syn]
    test_y = np.array([1]*test_real.shape[0] + [0]*test_syn.shape[0])

    return train_x, test_x, train_y, test_y


def discriminative_score(ori_data, syn_data):
    gpu = "cuda:0"
    batch_size = 128
    iteration = 500
    learning_rate = 0.01

    dis_net = Discriminator().to(gpu)
    train_x, test_x, train_y, test_y = train_test_devide(ori_data,syn_data)
    _, S, C =train_x.shape
    train_x = torch.from_numpy(train_x).contiguous().view(-1,C,1,S).to(torch.float32).to(gpu)
    train_y = torch.from_numpy(train_y).to(torch.float32).to(gpu)
    test_x = torch.from_numpy(test_x).contiguous().view(-1,C,1,S).to(torch.float32).to(gpu)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),learning_rate)
    
    # train
    for i in range(iteration):
        # shuffle
        idx = np.arange(len(train_y))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]
        loss_avg = []

        for j in range(len(train_y)//batch_size):
            data = train_x[batch_size*j : batch_size*(j+1)]
            label = train_y[batch_size*j : batch_size*(j+1)]

            y_pred = dis_net(data) 
            loss = loss_fn(y_pred.flatten(),label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_avg.append(loss.item())
        loss_avg = np.mean(loss_avg)
        if i % 200 == 0:
            print(f"loss: {loss_avg:>7f}  [{i:>5d}/{iteration:>5d}]")
    
    # test
    dis_net.eval()
    
    y_pred = dis_net(test_x).flatten()
    y_pred = nn.functional.sigmoid(y_pred).cpu().detach().numpy()
    y_pred = y_pred > 0.5
    acc = np.sum(y_pred==test_y) / len(test_y)
    
    return abs(0.5-acc)


if __name__ == '__main__':
    source = torch.tensor(np.random.randn(3,150))
    target = torch.tensor(np.random.randn(3,150))
    mmd = mmd_rbf(source,target)
    print('done')