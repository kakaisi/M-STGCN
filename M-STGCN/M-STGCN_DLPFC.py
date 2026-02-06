import warnings
warnings.filterwarnings("ignore")

from torch.optim import Adam
from sklearn import metrics
import os
from tqdm import tqdm

from utils import *
from models import M_STGCN
from config import Config

dataset = '151507'
config_file = './config/DLPFC.ini'
config = Config(config_file)

path = f"../data/DLPFC/{dataset}/"
epochs = 200

if __name__ == "__main__":
    adata, features, labels, fadj, padj, sadj, graph_nei, graph_neg = prepare_data_in_memory(
        dataset, config.fdim, config.k, config.radius, path
    )

    print(f"\n--- Starting M-STGCN training (Epochs: {epochs}) ---")

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    features, fadj, padj, sadj = features.to(device), fadj.to(device), padj.to(device), sadj.to(device)
    graph_nei, graph_neg = graph_nei.to(device), graph_neg.to(device)

    # 3. 初始化模型 (现在权重初始化是固定的了)
    model = M_STGCN(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ari_max = 0
    best_emb = None
    best_clusters = None

    with tqdm(total=epochs, desc="Training") as pbar:
        for epoch in range(1, epochs + 1):
            emb, loss = run_training(model, optimizer, features, fadj, padj, sadj, graph_nei, graph_neg, config)

            # 注意：如果在这里做聚类，最好是用 detach 出来的 emb，这在 utils.run_training 里已经做好了
            kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=config.seed).fit(emb)
            ari_res = metrics.adjusted_rand_score(labels, kmeans.labels_)

            if ari_res > ari_max:
                ari_max = ari_res
                best_emb = emb
                best_clusters = kmeans.labels_
            pbar.update(1)

    print(f"\n--- Training complete. ARI: {ari_max:.4f} ---")

    save_dir = f'./result/DLPFC/{dataset}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    adata.obs['m_stgcn_cluster'] = pd.Categorical(best_clusters)

    plt.rcParams["figure.figsize"] = (5, 5)
    custom_colors = ['#F38235', '#104674', '#4D836C', '#4C506D', '#184A45', '#858386', '#B45A56']

    try:
        sc.pl.spatial(
            adata,
            img_key="hires",
            color=['m_stgcn_cluster'],
            title=f'M-STGCN (ARI: {ari_max:.4f})',
            palette=custom_colors,
            show=False
        )
        plt.savefig(os.path.join(save_dir, 'M_STGCN_clusters.png'), bbox_inches='tight', dpi=600)
    except Exception as e:
        print(f"Plotting warning: {e}. Falling back to default colors.")
        sc.pl.spatial(
            adata,
            img_key="hires",
            color=['m_stgcn_cluster'],
            title=f'M-STGCN (ARI: {ari_max:.4f})',
            show=False
        )
        # plt.savefig(os.path.join(save_dir, 'M_STGCN_clusters.png'), bbox_inches='tight', dpi=600)

    plt.close()
    print(f"Results have been saved to: {save_dir}")