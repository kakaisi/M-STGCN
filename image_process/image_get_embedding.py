import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from byol_pytorch import BYOL
from PIL import Image
from tqdm import tqdm

torch.manual_seed(2025)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import numpy as np
random.seed(2025)
np.random.seed(2025)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = sorted(
            [x for x in os.listdir(root_dir) if x.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建数据集
def create_data_loader(data_path, batch_size=43):
    dataset = CustomDataset(root_dir=data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, data_loader

def initialize_byol_model():
    model = BYOL(
        models.resnet50(pretrained=True),
        image_size=256,
        hidden_layer='avgpool',
        augment_fn=None
    ).double()

    if torch.cuda.is_available():
        model = model.cuda()

    return model

# 提取图像特征并保存
def extract_features(model, data_loader, output_file='embeddings.npy'):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for images in tqdm(data_loader, desc='Extracting features'):
            images = images.cuda().double() if torch.cuda.is_available() else images.double()
            _, embedding = model(images, return_embedding=True)
            embeddings.append(embedding.cpu().numpy())


    embeddings = np.vstack(embeddings)
    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}, shape: {embeddings.shape}")


def main():
    # dataset_name = 'Human_Breast_Cancer'
    slices = ['151507']
    output_base_dir = f"../Data/DLPFC/"

    for slide in slices:
        data_dir = os.path.join(output_base_dir, slide, 'patch_image')
        output_file = os.path.join(output_base_dir, slide, 'embeddings.npy')

        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist, skipping.")
            continue


        dataset, data_loader = create_data_loader(data_dir)

        model = initialize_byol_model()

        extract_features(model, data_loader, output_file)

if __name__ == "__main__":
    main()
