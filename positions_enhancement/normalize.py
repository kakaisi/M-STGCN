from typing import Optional
from anndata import AnnData
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from ._weighting_matrix import (
    calculate_weight_matrix,
    impute_neighbour,
    _WEIGHTING_MATRIX,
    _PLATFORM,
)


def SME_normalize(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "physical_distance",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Adjust data using spatial location (S)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_data : str, default='raw'
        Input data, can be `raw` counts or log transformed data.
    weights : _WEIGHTING_MATRIX, default='physical_distance'
        Weighting matrix for imputation.
        Only supports `"physical_distance"`.
    platform : _PLATFORM, default='Visium'
        Data platform type, supports only `"Visium"`.
    copy : bool, default=False
        Return a copy instead of modifying the original `adata`.

    Returns
    -------
    AnnData, optional
        If `copy=True`, returns the modified `AnnData` object; otherwise, returns `None`, and modifies the original object.
    """
    if copy:
        adata = adata.copy()

    # Select data to normalize
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.DataFrame):
            count_embed = adata.X.values
        else:
            raise ValueError(
                f"""\
                    {type(adata.X)} is not a valid type.
                    """
            )
    else:
        if use_data not in adata.obsm.keys():
            raise ValueError(f"The key '{use_data}' does not exist in `adata.obsm`.")
        count_embed = adata.obsm[use_data]

    # Calculate physical distance weighting matrix
    calculate_weight_matrix(adata, platform=platform)

    # Perform neighborhood-based imputation based on physical distance
    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    # Retrieve imputed data and replace zeros with NaN
    imputed_data = adata.obsm["imputed_data"].astype(float)
    imputed_data[imputed_data == 0] = np.nan

    # Calculate normalized count matrix by averaging original and imputed data
    adjusted_count_matrix = np.nanmean(np.array([count_embed, imputed_data]), axis=0)

    # Add normalized data to AnnData object
    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjusted_count_matrix

    print(f"The data adjusted by SME is added to adata.obsm['{key_added}']")

    return adata if copy else None
