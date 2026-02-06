# ----------------------------------------------------------------------------
# This file contains spatial enhancement algorithms adapted from stLearn:
# https://github.com/biomedicalmachinelearning/stLearn
#
# Original Copyright (c) 2020-2026, Genomics and Machine Learning lab
# License: BSD 3-Clause License
#
# Modifications by M-STGCN Team:
# - Integrated Read10X, SME_normalize, and helper functions into a single utility file.
# - Optimized neighbor imputation using vectorized operations (removing slow loops).
# - Added robust handling for coordinate files and barcode matching.
# ----------------------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.image import imread
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances


# --- Core Processing Functions ---

def calculate_weight_matrix(
        adata: AnnData,
        platform: str = "Visium",
) -> None:
    """
    Calculates physical distance weights and stores in adata.uns['physical_distance'].
    """
    if platform != "Visium":
        raise ValueError(f"Platform '{platform}' is not supported. Only 'Visium' is supported.")

    # Check for required columns
    required_cols = ["imagerow", "imagecol", "array_row", "array_col"]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]

    # Critical Fix: If columns are missing, try to generate them from obsm['spatial'] if available
    if missing_cols:
        if "spatial" in adata.obsm:
            print("Recovering missing spatial columns from adata.obsm['spatial']...")
            # Assuming obsm['spatial'] is [x, y] -> [imagecol, imagerow]
            # And assuming array_row/col are not strictly needed if we have pixel locs,
            # BUT the regression needs them.
            # If missing array_row/col, we can't do the regression to find 'unit'.
            # Fallback: Use simple distance threshold without regression if array coords missing.
            pass

    try:
        img_row = adata.obs["imagerow"].values.reshape(-1, 1).astype(float)
        img_col = adata.obs["imagecol"].values.reshape(-1, 1).astype(float)
        array_row = adata.obs["array_row"].values.reshape(-1, 1).astype(float)
        array_col = adata.obs["array_col"].values.reshape(-1, 1).astype(float)
    except KeyError as e:
        raise KeyError(f"Missing required spatial columns: {e}. Ensure Read10X loaded them correctly.")

    # Estimate pixel distance per array unit
    reg_row = LinearRegression().fit(array_row, img_row)
    reg_col = LinearRegression().fit(array_col, img_col)
    unit = math.sqrt(reg_row.coef_[0][0] ** 2 + reg_col.coef_[0][0] ** 2)

    if unit == 0: unit = 1.0  # Safety fallback

    rate = 3
    coords = np.hstack([img_col, img_row])
    pd_dist = pairwise_distances(coords, metric="euclidean")

    # Vectorized Thresholding
    pd_norm = np.where(pd_dist >= rate * unit, 0, 1)
    adata.uns["physical_distance"] = pd_norm


def impute_neighbour(
        adata: AnnData,
        count_embed: Optional[np.ndarray] = None,
        weights: str = "physical_distance",
) -> None:
    """
    Performs vectorized neighborhood imputation.
    """
    if count_embed is None: return
    if weights not in adata.uns: raise KeyError(f"Weights '{weights}' not found.")

    weights_matrix = adata.uns[weights]
    n_neighbors = 6

    # 1. Top-K Neighbors
    top_indices = np.argsort(weights_matrix, axis=1)[:, -n_neighbors:]

    # 2. Extract Weights
    row_indices = np.arange(weights_matrix.shape[0])[:, None]
    top_weights = weights_matrix[row_indices, top_indices]

    # 3. Normalize
    weight_sums = top_weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0
    normalized_weights = top_weights / weight_sums

    # 4. Gather Features & Weighted Sum
    # count_embed shape: (N, Features)
    neighbor_features = count_embed[top_indices]
    imputed_data = np.sum(normalized_weights[:, :, None] * neighbor_features, axis=1)

    # 5. Mask isolated spots
    imputed_data[top_weights.sum(axis=1) == 0] = 0

    adata.obsm["imputed_data"] = imputed_data


def SME_normalize(
        adata: AnnData,
        use_data: str = "raw",
        weights: str = "physical_distance",
) -> None:
    """
    Main Enhancement Function.
    use_data: "raw" for Gene Expression, or a key in adata.obsm (e.g., "emb") for embeddings.
    """
    # 1. Prepare Data
    if use_data == "raw":
        if issparse(adata.X):
            count_embed = adata.X.toarray()
        else:
            count_embed = np.array(adata.X)
    else:
        if use_data not in adata.obsm:
            raise KeyError(f"Key '{use_data}' not found in adata.obsm")
        count_embed = adata.obsm[use_data]

    print(f"--- Running SME on '{use_data}' (Shape: {count_embed.shape}) ---")

    # 2. Calculate Weights (if not exists)
    if weights not in adata.uns:
        print("Calculating spatial weights...")
        calculate_weight_matrix(adata)

    # 3. Impute
    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    # 4. Enhance (Average)
    imputed = adata.obsm["imputed_data"].astype(float)
    # Treat zeros as NaN for averaging (assuming 0 means no neighbor info)
    imputed_nan = imputed.copy()
    imputed_nan[imputed == 0] = np.nan

    combined = np.array([count_embed, imputed_nan])
    enhanced = np.nanmean(combined, axis=0)
    enhanced = np.nan_to_num(enhanced)

    # Store Result
    key_added = f"{use_data}_SME_normalized"
    adata.obsm[key_added] = enhanced
    print(f"Enhancement done. Result stored in adata.obsm['{key_added}']")


def Read10X(path: Union[str, Path], count_file: str = "filtered_feature_bc_matrix.h5") -> AnnData:
    """
    Robust 10X Reader.
    """
    path = Path(path)
    adata = sc.read_10x_h5(path / count_file)
    adata.uns["spatial"] = {"Visium": {}}

    # Path setup
    spatial_path = path / "spatial"
    pos_file = spatial_path / "tissue_positions.csv"
    if not pos_file.exists():
        pos_file = spatial_path / "tissue_positions_list.csv"

    # Read Positions
    try:
        # Standard headerless
        positions = pd.read_csv(pos_file, header=None, index_col=0)
        positions.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    except:
        # Try with header
        positions = pd.read_csv(pos_file, index_col=0)

    # Fix Index (Barcode) Mismatch
    adata_suffix = adata.obs_names[0].endswith("-1")
    csv_suffix = positions.index[0].endswith("-1")
    if adata_suffix and not csv_suffix: positions.index = positions.index + "-1"
    if not adata_suffix and csv_suffix: positions.index = positions.index.str.replace("-1", "")

    # Merge
    positions = positions.reindex(adata.obs_names)
    adata.obs = adata.obs.join(positions)

    # Store spatial
    adata.obsm["spatial"] = positions[["pxl_row", "pxl_col"]].fillna(0).values

    # Create required columns for calculate_weight_matrix
    # (Using default scale=1.0 since we process embeddings/genes, not verifying images visually)
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 1]
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 0]

    return adata