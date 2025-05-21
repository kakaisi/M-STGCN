from sklearn.metrics import pairwise_distances
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from ._compat import Literal
from tqdm import tqdm

# Define supported platform type
_PLATFORM = Literal["Visium"]

# Define supported weighting matrix type, retaining only "physical_distance"
_WEIGHTING_MATRIX = Literal["physical_distance"]


def calculate_weight_matrix(
    adata: AnnData,
    adata_imputed: Union[AnnData, None] = None,
    pseudo_spots: bool = False,
    platform: _PLATFORM = "Visium",
) -> Optional[AnnData]:
    """
    Calculate the physical distance matrix and store it in adata.uns.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    adata_imputed : AnnData, optional
        If there is an imputed AnnData object, pass it here (currently not used).
    pseudo_spots : bool, optional
        Whether to handle pseudo spots (currently retained but only processing "physical_distance").
    platform : _PLATFORM
        Data platform type, supports "Visium".

    Returns
    -------
    AnnData, optional
        If applicable, returns the modified AnnData object.
    """
    from sklearn.linear_model import LinearRegression
    import math

    # Extract spatial coordinates and set scaling factor based on platform type
    if platform == "Visium":
        img_row = adata.obs["imagerow"].astype(float)
        img_col = adata.obs["imagecol"].astype(float)
        array_row = adata.obs["array_row"].astype(float)
        array_col = adata.obs["array_col"].astype(float)
        rate = 3
    else:
        raise ValueError(f"{platform!r} is not supported.")

    # Fit linear regression models to estimate the pixel distance per array unit
    reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
    reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)

    if pseudo_spots and adata_imputed:
        # Calculate physical distances between imputed data and original data
        pd = pairwise_distances(
            adata_imputed.obs[["imagecol", "imagerow"]],
            adata.obs[["imagecol", "imagerow"]],
            metric="euclidean",
        )
        unit = math.sqrt(reg_row.coef_[0] ** 2 + reg_col.coef_[0] ** 2)
        pd_norm = np.where(pd >= unit, 0, 1)

        # Store physical distance in adata_imputed
        adata_imputed.uns["physical_distance"] = pd_norm

    else:
        # Calculate physical distances within original data
        pd = pairwise_distances(adata.obs[["imagecol", "imagerow"]], metric="euclidean")
        unit = math.sqrt(reg_row.coef_[0] ** 2 + reg_col.coef_[0] ** 2)
        pd_norm = np.where(pd >= rate * unit, 0, 1)

        # Store physical distance in adata.uns
        adata.uns["physical_distance"] = pd_norm

    return adata


def impute_neighbour(
    adata: AnnData,
    count_embed: Union[np.ndarray, None] = None,
    weights: _WEIGHTING_MATRIX = "physical_distance",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Perform neighborhood-based weighted imputation based on physical distance.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    count_embed : np.ndarray, optional
        The data embedding matrix to impute. If None, no imputation is performed.
    weights : _WEIGHTING_MATRIX
        The type of weighting matrix to use, "physical_distance".
    copy : bool, optional
        Whether to return a copy instead of modifying the original adata.

    Returns
    -------
    AnnData, optional
        If copy=True, returns the modified AnnData object; otherwise, returns None.
    """
    coor = adata.obs[["imagecol", "imagerow"]]
    weights_matrix = adata.uns[weights]

    lag_coor = []
    weights_list = []

    with tqdm(
        total=len(adata),
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(len(coor)):
            main_weights = weights_matrix[i]
            # For "physical_distance", select the top 6 neighbors with highest weights
            current_neighbour = main_weights.argsort()[-6:]

            surrounding_count = count_embed[current_neighbour]
            surrounding_weights = main_weights[current_neighbour]
            if surrounding_weights.sum() > 0:
                surrounding_weights_scaled = surrounding_weights / surrounding_weights.sum()
                weights_list.append(surrounding_weights_scaled)

                surrounding_count_adjusted = surrounding_weights_scaled.reshape(-1, 1) * surrounding_count
                surrounding_count_final = np.sum(surrounding_count_adjusted, axis=0)
            else:
                surrounding_count_final = np.zeros(count_embed.shape[1])
                weights_list.append(np.zeros(len(current_neighbour)))

            lag_coor.append(surrounding_count_final)
            pbar.update(1)

    imputed_data = np.array(lag_coor)
    key_added = "imputed_data"
    adata.obsm[key_added] = imputed_data

    adata.obsm["top_weights"] = np.array(weights_list)

    return adata if copy else None
