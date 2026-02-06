# ----------------------------------------------------------------------------
# Image processing pipeline adapted from STAIG:
# https://github.com/y-itao/STAIG
# ----------------------------------------------------------------------------

import os
import numpy as np
import torch
from image_processor import SpatialImageProcessor


def run_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)
    # Note: Adjust data root as needed for your specific environment
    data_root = os.path.join(project_root, "data", "DLPFC", "151507")

    processor = SpatialImageProcessor(patch_size=500, target_size=256)

    adata = processor.load_adata(data_root)

    patch_path = os.path.join(data_root, "patch_image")
    processor.extract_patches(adata,
                              full_image_path=os.path.join(data_root, "spatial", "tissue_full_image.tif"),
                              output_dir=patch_path)

    filter_path = os.path.join(data_root, "patch_image_filter")
    processor.apply_fft_filter(input_dir=patch_path, output_dir=filter_path)

    print("Starting GPU Feature Extraction...")
    embeddings = processor.get_embeddings(filter_path, batch_size=64)

    save_path = os.path.join(data_root, "embeddings.npy")
    np.save(save_path, embeddings)
    print(f"Successfully saved embeddings to {save_path}, shape: {embeddings.shape}")


if __name__ == "__main__":
    run_pipeline()