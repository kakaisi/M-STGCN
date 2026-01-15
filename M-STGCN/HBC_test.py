from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from models import M_STGCN
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings

# 忽略 FutureWarning 和 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 你的代码逻辑
from sklearn.cluster import KMeans

kmeans = KMeans()  # 示例代码，正常调用 KMeans


def load_data(dataset):
    print("load data:")
    path = "../generate_data/" + dataset + "/M-STGCN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    padj = adata.obsm["padj"]
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    npadj = normalize_sparse_matrix(padj + sp.eye(padj.shape[0]))
    npadj = sparse_mx_to_torch_sparse_tensor(npadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, npadj, graph_nei, graph_neg


def train():
    model.train()
    optimizer.zero_grad()
    # com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    emb, pi, disp, mean = model(features, sadj, fadj, padj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    # con_loss = consistency_loss(com1, com2)
    # total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss,  total_loss

# custom_colors = [
#     '#4B2991', '#BB3754', '#007856', '#FFA500', '#0072B2',
#     '#654522', '#56B4E9', '#009E73', '#F0E442', '#D55E00',
#     '#CC79A7', '#666666', '#E69F00', '#2D9F9F', '#999933',
#     '#8673CF', '#ED645A', '#44AA99', '#DDCC77', '#117733'
#                 ]
custom_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
]


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Human_Breast_Cancer']

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = '../result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, labels, fadj, sadj, padj, graph_nei, graph_neg = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = '../result/Human_Breast_Cancer/'
        plt.rcParams["figure.figsize"] = (3, 3)

        print(adata)
        title = "Manual annotation"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], palette=custom_colors, title=title, show=False)
        plt.savefig(savepath + dataset + '.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            padj = padj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = 500
        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = M_STGCN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, total_loss = train()
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_

            ari_res = metrics.adjusted_rand_score(labels, idx)
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
                
        title = 'M-STGCN: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'],palette=custom_colors, title=title, show=False)
        plt.savefig(savepath + 'M-STGCN.pdf', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                           show=False)
        # plt.savefig(savepath + 'M-STGCN_umap_mean.pdf', bbox_inches='tight', dpi=600)
        plt.show()

        # pd.DataFrame(emb_max).to_csv(savepath + 'M-STGCN_emb.csv')
        # pd.DataFrame(idx_max).to_csv(savepath + 'M-STGCN_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'M-STGCN.h5ad')
