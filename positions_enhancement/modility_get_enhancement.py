import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
# import module
from pathlib import Path
from positions_enhancement.normalize import SME_normalize
from positions_enhancement.read import Read10X

# specify PATH to data
BASE_PATH = Path("../Data/DLPFC/")
sample_list = ["151507"]
# sample_list = ['151508', '151509', '151510', '151669', "151670",
#                '151671', '151672', '151673', '151674', '151675', '151676']

for i in range(1):
    sample = sample_list[i]
    print('{} is processing'.format(sample))
    GROUND_TRUTH_PATH = BASE_PATH / sample / (sample + "_truth.txt")

    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH, sep='\t', header=None, index_col=0)
    ground_truth_df.columns = ["ground_truth"]

    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
    ground_truth_df["ground_truth_le"] = ground_truth_le

    # data = st.Read10X(BASE_PATH / sample)
    data = Read10X(BASE_PATH / sample)
    data.var_names_make_unique()
    ground_truth_df = ground_truth_df.reindex(data.obs_names)
    data.obs["ground_truth"] = pd.Categorical(ground_truth_df["ground_truth"])
    data.obs['ground_truth'].isna().sum()
    data = data[~data.obs['ground_truth'].isna()]

    n_cluster = len(set(ground_truth_df["ground_truth"])) - 1
    data.obs['ground_truth'] = ground_truth_df["ground_truth"]
    ground_truth_le = ground_truth_df["ground_truth_le"]

    sc.pp.highly_variable_genes(data, flavor="seurat_v3", n_top_genes=3000)
    data = data[:, data.var['highly_variable']]

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.pca(data, n_comps=50)

    SME_normalize(data, use_data="raw", weights="physical_distance")
    gene_positions = data.obsm['raw_SME_normalized']
    # np.save(f"D:/Pycharm Project/own\positions_enhancement/output/{sample}_gene_positions.npy", gene_positions)
