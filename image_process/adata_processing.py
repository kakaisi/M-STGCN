import scanpy as sc
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import pandas as pd
import os


class LoadSingle10xAdata:
    def __init__(self, path: str, n_top_genes: int = 3000, image_emb: bool = False,
                 label: bool = True, filter_na: bool = True):
        """
        Initialize the LoadSingle10xAdata class.
        Parameters:
            path (str): Path to the data directory.
            n_top_genes (int): Number of top highly variable genes to select, default is 3000.
            image_emb (bool): Whether to load image embeddings, default is False.
            label (bool): Whether to load label information, default is True.
            filter_na (bool): Whether to filter out missing labels, default is True.
        """
        self.path = path
        self.n_top_genes = n_top_genes
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.adata = None

    def load_data(self):
        """
        Load 10x Genomics Visium spatial transcriptomics data.
        """
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()

    def load_label(self):
        """
        Load and process label information.
        """
        label_file = os.path.join(self.path, 'truth.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file '{label_file}' not found.")

        df_meta = pd.read_csv(label_file, sep='\t', header=None)
        if df_meta.shape[1] < 2:
            raise ValueError(f"Label file '{label_file}' format is incorrect, requires at least two columns.")

        df_meta_layer = df_meta[1]
        self.adata.obs['ground_truth'] = df_meta_layer.values

        if self.filter_na:
            initial_count = self.adata.n_obs
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]
            filtered_count = self.adata.n_obs
            print(f"Filtered out {initial_count - filtered_count} cells with missing labels.")

    # The following methods are not currently used and have been commented out
    """
    def preprocess(self):
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)

    def construct_interaction(self, n_neighbors: int = 3):
        position = self.adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)

        self.adata.obsm['graph_neigh'] = adj

    def generate_gene_expr(self):
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        if isinstance(adata_Vars.X, (csc_matrix, csr_matrix)):
            feat = adata_Vars.X.toarray()
        else:
            feat = adata_Vars.X

        self.adata.obsm['feat'] = feat

    def load_image_emb(self):
        data = np.load(os.path.join(self.path, 'embeddings_gray.npy'))
        data = data.reshape(data.shape[0], -1)
        scaler = StandardScaler()
        embedding = scaler.fit_transform(data)
        pca = PCA(n_components=16, random_state=42)
        embedding = pca.fit_transform(embedding)
        self.adata.obsm['img_emb'] = embedding
        pca_g = PCA(n_components=64, random_state=42)
        self.adata.obsm['feat_pca'] = pca_g.fit_transform(self.adata.obsm['feat'])
        self.adata.obsm['con_feat'] = np.concatenate([self.adata.obsm['feat_pca'], self.adata.obsm['img_emb']], axis=1)
        con_feat = self.adata.obsm['con_feat']

        scaler = StandardScaler()
        con_feat_standardized = scaler.fit_transform(con_feat)
        self.adata.obsm['con_feat'] = con_feat_standardized

    def calculate_edge_weights(self):
        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = np.log(edge_weights[i][non_zero_indices])
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities

        if self.kernel == 'rbf':
            gamma = 0.01
            similarity_matrix = rbf_kernel(node_emb, gamma=gamma)
            edge_weights = np.where(graph_neigh == 1, 1 - similarity_matrix, 0)

            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

            self.adata.obsm['edge_probabilities'] = edge_probabilities

        if self.kernel == 'cosine':
            euclidean_distances = cdist(node_emb, node_emb, metric='cosine')
            edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)

            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

            self.adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):
        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities
    """

    # If you need to run all steps, you can uncomment the following method
    """
    def run(self):
        self.load_data()
        if self.label:
            self.load_label()
        self.preprocess()
        self.construct_interaction()
        self.generate_gene_expr()

        if self.image_emb:
            self.load_image_emb()
            self.calculate_edge_weights()
        else:
            self.calculate_edge_weights_gene()

        print('adata load done')
        return self.adata
    """
