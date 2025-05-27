# Spatially Contrastive Variational Autoencoder for Spatial Transcriptomics Analysis

## Overview

**M-STGCN** (Multimodal Spatio-Temporal Graph Convolutional Network) is a novel unsupervised framework designed for **spatial domain identification** and **multimodal data integration** in spatially resolved transcriptomics (SRT). By synergistically combining gene expression, spatial coordinates, and histopathological images, M-STGCN enables:

- **Context-aware representation learning** through graph attention mechanisms  
- **Noise-robust denoising** via zero-inflated negative binomial (ZINB) decoding  
- **Cross-modal alignment** of molecular and morphological features  



![image](https://github.com/user-attachments/assets/1b0d2b76-1944-4d61-9d3d-169c6da04f7e)


## Requirements

### Hardware
- GPU: NVIDIA GeForce RTX 4060ti(or higher with CUDA support)  
- CUDA Version: 12.3 

### Software
```python
python==3.8
torch==2.0.0
numpy==1.21.6
scanpy==1.9.3
anndata==0.8.0
scikit-learn==1.2.2
tqdm==4.65.0
matplotlib==3.7.1
