import cv2
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from image_process.adata_processing import LoadSingle10xAdata

# Configuration settings
root_data_dir = Path("D:/Pycharm Project/M-STGCN(github)/Data/DLPFC")  # Root directory
patch_size = int(3.5 * 145)  # Calculate patch size
patch_image_base_path = "patch_image"  # Directory to store the extracted patches
output_base_path = "patch_image_filter"  # Directory to store the filtered images

# Define the list of slices you want to process manually
# Example: ["151508", "151509", "151510"]
slices_to_process = ['151507']  # Customize this list to include your desired slices


# Helper function to create a directory if not exists
def create_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{directory.name}' {'created' if directory.exists() else 'already exists'}")


# Function to read and process the image patches
def process_image(filename, input_path, output_path, GaussianBlur, lower, upper):
    image_path = input_path / filename
    image = cv2.imread(str(image_path), 0)  # Read as grayscale

    # Resize if necessa ry
    original_height, original_width = image.shape
    if (original_height, original_width) != (512, 512):
        image = cv2.resize(image, (512, 512))

    if GaussianBlur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    custom_mask = create_custom_mask(image.shape, lower, lower, upper, upper)
    fshift_masked = fshift * custom_mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    image_filtered = np.abs(np.fft.ifft2(f_ishift))

    if GaussianBlur:
        image_filtered = cv2.GaussianBlur(image_filtered, (15, 15), 0)

    # Convert to RGB and resize to 224x224
    image_filtered_rgb = cv2.cvtColor(np.float32(image_filtered), cv2.COLOR_GRAY2RGB)
    mae_patch = cv2.resize(image_filtered_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Save the filtered image in original size
    if (original_height, original_width) != (512, 512):
        image_filtered_rgb = cv2.resize(image_filtered_rgb, (original_width, original_height))

    output_file = output_path / filename

    # Check if the output file already exists, if so, skip the processing
    if output_file.exists():
        print(f"File {output_file.name} already exists, skipping.")
        return mae_patch

    cv2.imwrite(str(output_file), image_filtered_rgb)
    return mae_patch


# Function to create a custom mask for frequency filtering
def create_custom_mask(image_shape, x1, y1, x2, y2):
    mask = np.zeros(image_shape, np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


# Function to process each directory
def process_directory(subdir):
    data_dir = root_data_dir / subdir  # Full path to the current subdirectory
    patch_image_path = data_dir / patch_image_base_path
    output_path = data_dir / output_base_path

    # Create directories if not exist
    create_dir(patch_image_path)
    create_dir(output_path)

    # Load data and set up patch extraction
    loader = LoadSingle10xAdata(path=str(data_dir), image_emb=False, label=True, filter_na=True)
    loader.load_data()
    adata = loader.adata

    # Check if the full image exists
    full_image_path = data_dir / "spatial" / "tissue_full_image.tif"
    if full_image_path.exists():
        print(f"Processing full image in {subdir}")
    else:
        print(f"Full image not found in {subdir}, skipping.")
        return

    # Read full image
    im = cv2.imread(str(full_image_path), cv2.IMREAD_COLOR)

    # Process and save image patches
    for i, coord in tqdm(enumerate(adata.obsm['spatial']), total=len(adata.obsm['spatial'])):
        left = int(coord[0] - patch_size / 2)
        top = int(coord[1] - patch_size / 2)
        right = left + patch_size
        bottom = top + patch_size

        patch = im[top:bottom, left:right]
        resized_patch = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR) if patch_size != 512 else patch
        cv2.imwrite(str(patch_image_path / f'{i}.png'), resized_patch)

    # Filter image patches using multiprocessing
    filter_patches_using_multiprocessing(patch_image_path, output_path)


# Function to filter image patches using multiprocessing
def filter_patches_using_multiprocessing(patch_image_path, output_path):
    png_files = [name for name in os.listdir(patch_image_path) if name.endswith('.png')]
    args = [(filename, patch_image_path, output_path, True, 245, 275) for filename in png_files]

    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_image, args)


# Main processing loop
if __name__ == "__main__":
    # Process only the slices you define in the list `slices_to_process`
    for subdir in slices_to_process:
        process_directory(subdir)
