# M-STGCN: image-augmented graph convolutional framework for joint spatial domain identification and gene expression denoising 
> [!IMPORTANT]
> **🚧 Maintenance Notice / 维护公告**
>
> Since the paper has been officially published, we are currently refactoring the code to include standard **License** agreements and detailed **Citations** for the algorithms and datasets used.
>
> The core functionality (M-STGCN model) is fully operational, but some file structures or comments are being optimized. The complete, polished version (v1.1) will be updated shortly.
>
> *论文已见刊，代码仓库正在进行规范化维护（补充开源协议与参考文献引用）。核心功能可用，完整更新版即将发布。*

<img width="2238" height="2054" alt="figure 1" src="https://github.com/user-attachments/assets/38d13c21-d52c-4dc6-8c95-85dbf1756d33" />



## Requirements

### Hardware
- Recommended GPU: NVIDIA GeForce RTX 4060ti or higher (CUDA support required)
- CUDA Version: 12.3 

### Software
- Python
- numpy
- pandas
- scipy
- pytorch
- torch_geometric
- torch_sparse
- torch_scatter
- matplotlib
- R
- rpy2

## Usage Tutorial

**1、Morphology feature extraction**
- Preprocess H&E stained images and extract image patches
- Use pre-trained models to extract feature vectors for each patch
  
**Execution Command**

Run **image_process/image_get_patches.py** to preprocess the H&E image:

python image_get_patches.py

Then,run **image_process/image_get_embedding.py** to exact the feature of H&E image:

python image_get_embedding.py


**2、Data augmentation**

Run **positions_enhancement/modality_get_enhancement.py** to augament the gene expression and image feature.

python modility_get_enhancement.py

The augmented data  will be saved in the  **positions_enhancement/output** 

**3、Integrate modality via M-STGCN**

First,we run the **M-STGCN/_generate_data.py** to construct graphs

python DLPFC_generate_data.py

Then,input the generate_data to **M-STGCN/_test_data.py**

python DLPFC_test_data.py

ALL results are saved in the result folder.Such as **M-STGCN_idx.csv**(cell clustering label)、 **M-STGCN_emb.csv**(Multimodal Latent Embeddings).
