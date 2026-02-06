# ----------------------------------------------------------------------------
# This file contains code adapted from the STAIG repository:
# https://github.com/y-itao/STAIG
#
# Original Author: y-itao
# License: Apache License 2.0
#
# Modifications by M-STGCN Team:
# - Adapted image patch extraction and FFT filtering logic.
# - Integrated GPU-accelerated embedding extraction.
# ----------------------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from byol_pytorch import BYOL


def set_seed(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SpatialImageProcessor:
    """
    Processor for extracting image features from spatial transcriptomics data.
    Logic for patch extraction and FFT filtering is adapted from STAIG.
    """
    def __init__(self, patch_size=512, target_size=256, device=None):
        self.patch_size = patch_size
        self.target_size = target_size
        # Prioritize GPU usage
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        set_seed()

    def load_adata(self, data_path):
        """Load 10x Visium data."""
        data_path = Path(data_path)
        adata = sc.read_visium(str(data_path), count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()

        label_file = data_path / '151507_truth.txt'
        if label_file.exists():
            df_meta = pd.read_csv(label_file, sep='\t', header=None)
            adata.obs['ground_truth'] = df_meta[1].values
            adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
        return adata

    def extract_patches(self, adata, full_image_path, output_dir):
        """Extract image patches."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(full_image_path))
        if img is None: raise FileNotFoundError(f"Image not found at {full_image_path}")

        print(f"Extracting patches to {output_dir}...")
        for i, coord in tqdm(enumerate(adata.obsm['spatial']), total=adata.n_obs):
            x, y = int(coord[0]), int(coord[1])
            left, top = x - self.patch_size // 2, y - self.patch_size // 2
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            if patch.size > 0:
                patch = cv2.resize(patch, (self.target_size, self.target_size))
                cv2.imwrite(str(output_dir / f"{i}.png"), patch)

    @staticmethod
    def _fft_filter_task(args):
        img_path, out_path, low, high = args
        img = cv2.imread(str(img_path), 0)
        if img is None: return
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        mask = np.zeros(img.shape, np.uint8)
        mask[low:high, low:high] = 1
        fshift_masked = fshift * mask
        img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_masked)))
        img_rgb = cv2.cvtColor(np.float32(img_filtered), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(out_path), img_rgb)

    def apply_fft_filter(self, input_dir, output_dir, low=245, high=275):
        """Multi-process frequency filtering."""
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        task_args = [(input_dir / f, output_dir / f, low, high) for f in files]
        print(f"Filtering {len(files)} patches...")
        with Pool(os.cpu_count()) as pool:
            pool.map(self._fft_filter_task, task_args)

    def get_embeddings(self, patch_dir, batch_size=64):
        """GPU-accelerated feature extraction."""
        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = PatchDataset(patch_dir, transform=transform)
        # pin_memory=True significantly improves GPU data transfer speed
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

        # Load ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = BYOL(resnet, image_size=self.target_size, hidden_layer='avgpool')

        # Key: Move to GPU and set to float32
        model = model.to(self.device).float()
        model.eval()

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding on {self.device}"):
                batch = batch.to(self.device).float()
                _, emb = model(batch, return_embedding=True)
                embeddings.append(emb.cpu().numpy())

        # Release GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.vstack(embeddings)


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = sorted(list(self.root_dir.glob('*.png')), key=lambda x: int(x.stem))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img