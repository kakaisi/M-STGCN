# M-STGCN
## M-STGCN: A Position-Aware Multimodal Graph Convolutional Framework for Joint Spatial Domain Identification and Gene Expression Denoising

## 📝 Description
Spatial transcriptomics (ST) has revolutionized our ability to profile gene expression within the architectural complexity of tissue microenvironments. However, decoding spatial heterogeneity requires robust multimodal data integration that unifies gene expression, spatial positions, and histopathological images to overcome modality-specific biases. Here, we propose **M-STGCN**, a multimodal unsupervised framework that constructs a spatial weight matrix from spatial coordinates to simultaneously refine both gene expression profiles and image features, by which we establish position-aware feature enhancement served as a core innovation before graph fusion. Verified on human brain and breast cancer datasets, M-STGCN significantly improves the accuracy of spatial domain identification. For ST platforms lacking images and at diverse resolutions, M-STGCN maintains robust performance utilizing only gene expression and spatial coordinates. By effectively denoising raw spatial transcriptomic profiles, our approach identifies more significant spatial domain marker genes, as well as potential prognostic biomarkers for breast cancer.

---

## 🖼️ Model Architecture
<img width="2238" height="2054" alt="figure 1" src="https://github.com/user-attachments/assets/f4763c5c-8ad3-466e-8448-b2aec9bd9bd0" />

---

## 🛠️ Requirements

To ensure reproducibility, we recommend using **Python 3.8**. The following specific versions are required:
Python == 3.8

numpy == 1.20.0

pandas == 1.4.4

scipy == 1.8.1

stlearn == 0.4.8

pytorch == 1.11.0

torch_geometric == 2.1.0

torch_sparse == 0.6.15

torch_scatter == 2.0.9

matplotlib == 3.5.3

opencv

scanpy

tqdm

## 🚀 Usage
Please follow the pipeline below to run the M-STGCN framework:
Example on **DLPFC**:
1. Image Feature Extraction
Run the following script to extract initial features from histology images
```bash
python image_process/main.py
```

2. Position-Aware Feature Enhancement
Run the following script to enhance both image features and gene expression features using spatial coordinate information:
```bash
python positions_enhancement/run_enhancement.py
```

3. Model Training & Domain Identification
Finally, execute the main model script (specifically for the DLPFC dataset):
```bash
python M-STGCN_DLPFC.py
```

## 🖊️ Citation
If you find this work useful, please cite our paper:Chen, Xin et al. “M-STGCN: A Position-Aware Multimodal Graph Convolutional Framework for Joint Spatial Domain Identification and Gene Expression Denoising.” ***Small methods***, e01812. 21 Jan. 2026, doi:10.1002/smtd.202501812.


