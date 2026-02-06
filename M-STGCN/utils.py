import scipy.sparse as sp
import sklearn
import torch
import networkx as nx
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import anndata as ad

EPS = 1e-15


# --- Loss Functions ---

def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


# --- Graph Construction Utils ---

def spatial_construct_graph1(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A = np.zeros((coor.shape[0], coor.shape[0]))

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def normalize_sparse_matrix(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# --- High Level Utils (Newly Added for Main Script) ---

def prepare_data_in_memory(dataset, fdim, k, radius, path=None):
    """
    Load data, perform enhancement integration, construct graphs, and return tensors.
    """
    if path is None:
        path = f"../data/DLPFC/{dataset}/"

    print(f"--- Preparing data for {dataset} ---")
    base_path = Path(path)

    # 1. Paths
    kegg_path = base_path / f"{dataset}_gene_enhanced.npy"
    image_path = base_path / f"{dataset}_emb_enhanced.npy"
    labels_path = base_path / "metadata.tsv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Metadata not found at {labels_path}")

    # 2. Load Truth
    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace({'WM': '0', 'Layer1': '1', 'Layer2': '2', 'Layer3': '3',
                    'Layer4': '4', 'Layer5': '5', 'Layer6': '6'}, inplace=True)

    # 3. Load Visium
    adata1 = sc.read_visium(base_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    # 4. Filter
    data_mat = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    # 5. Reconstruct AnnData
    adata = ad.AnnData(pd.DataFrame(data_mat, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))
    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.uns['spatial'] = adata1.uns['spatial']

    # 6. Normalize
    print("Selecting HVGs...")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # 7. Construct Graphs
    print("Constructing graphs using enhanced features...")

    # Image Graph (padj)
    if image_path.exists():
        pfea = np.load(image_path).astype('float32')
        if pfea.shape[0] == adata1.n_obs:
            pfea = np.delete(pfea, NA_labels, axis=0)
        padj = features_construct_graph(pfea, k=k)
    else:
        raise FileNotFoundError(f"Enhanced embeddings not found: {image_path}")

    # Gene Graph (fadj) and Features
    if kegg_path.exists():
        gene_enhanced = np.load(kegg_path)
        if gene_enhanced.shape[0] == adata1.n_obs:
            gene_enhanced = np.delete(gene_enhanced, NA_labels, axis=0)
        adata.X = gene_enhanced  # Update features to enhanced version
        fadj = features_construct_graph(adata.X, k=k)
    else:
        raise FileNotFoundError(f"Enhanced genes not found: {kegg_path}")

    # Spatial Graph (sadj)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    # 8. Convert to Tensors
    print("Converting to Tensors...")
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']

    # Normalize Adjacency Matrices
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    npadj = normalize_sparse_matrix(padj + sp.eye(padj.shape[0]))
    npadj = sparse_mx_to_torch_sparse_tensor(npadj)

    return adata, features, labels, nfadj, npadj, nsadj, graph_nei, graph_neg


def run_training(model, optimizer, features, fadj, padj, sadj, graph_nei, graph_neg, config):
    """
    Performs one epoch of training for M-STGCN.
    Returns embedding (numpy) and loss (float).
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    emb, pi, disp, mean = model(features, sadj, fadj, padj)

    # Loss calculation
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss

    total_loss.backward()
    optimizer.step()

    return emb.detach().cpu().numpy(), total_loss.item()