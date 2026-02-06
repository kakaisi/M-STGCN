import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from st_utils import Read10X, SME_normalize

# --- Configuration ---
BASE_PATH = Path(r"D:\M-STGCN\data\DLPFC")
SAMPLE_LIST = ["151507"]


def process_sample(sample_id: str):
    print(f"\n{'=' * 10} Processing Sample: {sample_id} {'=' * 10}")
    sample_dir = BASE_PATH / sample_id

    # 1. Read gene expression data
    print("1. Reading Gene Expression Data...")
    try:
        adata = Read10X(sample_dir)
        adata.var_names_make_unique()
        print(f"   - Initial spots: {adata.n_obs}")
    except Exception as e:
        print(f"Error reading 10X data: {e}")
        return

    # 2. Load Ground Truth and align (Critical fix step)
    # Note: embeddings.npy is usually generated based on spots with valid Truth,
    # so adata must be filtered first.
    truth_path = sample_dir / f"{sample_id}_truth.txt"
    if truth_path.exists():
        print(f"2. Loading Ground Truth from {truth_path.name}...")
        ground_truth = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
        ground_truth.columns = ["label"]

        # Merge Truth information into adata
        # Use reindex to ensure consistent order; missing values are filled with NaN
        ground_truth = ground_truth.reindex(adata.obs_names)
        adata.obs["ground_truth"] = ground_truth["label"]

        # Filter out spots without Ground Truth (This is the reason for 4226 -> 4221)
        valid_mask = ~adata.obs["ground_truth"].isna()
        n_dropped = adata.n_obs - valid_mask.sum()

        if n_dropped > 0:
            print(f"   - Filtering: Dropping {n_dropped} spots without Ground Truth labels.")
            adata = adata[valid_mask].copy()
            print(f"   - Remaining spots: {adata.n_obs}")
        else:
            print("   - All spots have Ground Truth labels.")

    else:
        print("Warning: Ground Truth file not found. Skipping filtering (Embeddings might mismatch!).")

    # 3. Load Image Embeddings
    emb_path = sample_dir / "embeddings.npy"
    if emb_path.exists():
        print(f"3. Loading Image Embeddings from {emb_path.name}...")
        embeddings = np.load(emb_path)

        # Dimensionality check
        if embeddings.shape[0] != adata.n_obs:
            print(f"Error: Embedding dim {embeddings.shape[0]} != Adata dim {adata.n_obs}")
            print("   - Hint: Check if embeddings were generated on a different subset of spots.")
            # If the difference is small, we can try truncation (as a last resort, but not recommended)
            if abs(embeddings.shape[0] - adata.n_obs) < 50 and embeddings.shape[0] < adata.n_obs:
                print(
                    "   !!! Critical Warning: Strict alignment failed. Trying to intersect by index is impossible for numpy arrays.")
                return
            else:
                return

        # Store embeddings in obsm
        adata.obsm['emb'] = embeddings
        print("   - Embeddings loaded successfully.")
    else:
        print("Warning: embeddings.npy not found!")
        # If no embedding, decide whether to continue based on needs;
        # here we choose to proceed with gene enhancement
        pass

    # 4. Preprocessing (Scanpy standard: HVG -> Log1p -> PCA)
    print("4. Preprocessing (HVG -> Log1p -> PCA)...")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=50)

    # 5. Execute Enhancement (SME Enhancement)
    print("5. Running SME Enhancement...")

    # A. Enhance gene expression data
    # Result stored in adata.obsm['raw_SME_normalized']
    SME_normalize(adata, use_data="raw", weights="physical_distance")

    # B. Enhance Image Embeddings
    # Result stored in adata.obsm['emb_SME_normalized']
    if 'emb' in adata.obsm:
        SME_normalize(adata, use_data="emb", weights="physical_distance")

    # 6. Save results
    print(f"6. Saving results to {sample_dir}...")

    # Save enhanced gene data
    if 'raw_SME_normalized' in adata.obsm:
        gene_enhanced = adata.obsm['raw_SME_normalized']
        gene_out_path = sample_dir / f"{sample_id}_gene_enhanced.npy"
        np.save(gene_out_path, gene_enhanced)
        print(f"   - Saved Genes: {gene_out_path.name} {gene_enhanced.shape}")

    # Save enhanced Embeddings
    if 'emb_SME_normalized' in adata.obsm:
        emb_enhanced = adata.obsm['emb_SME_normalized']
        emb_out_path = sample_dir / f"{sample_id}_emb_enhanced.npy"
        np.save(emb_out_path, emb_enhanced)
        print(f"   - Saved Embeddings: {emb_out_path.name} {emb_enhanced.shape}")


if __name__ == "__main__":
    for sample in SAMPLE_LIST:
        process_sample(sample)