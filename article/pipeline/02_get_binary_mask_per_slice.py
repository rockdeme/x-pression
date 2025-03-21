import argparse
import os
import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu

def process_single_slice(input_tiff_path, slice_index, patch_x=128, patch_y=128, stride=16, output_dir="output_slices"):
    """
    Process a single slice from a multi-slice TIFF file and save the results.

    Parameters:
        input_tiff_path (str): Path to the input TIFF file.
        slice_index (int): Index of the slice to process.
        patch_x (int): Patch width.
        patch_y (int): Patch height.
        stride (int): Stride for the sliding window.
        output_dir (str): Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the specified slice
    with tiff.TiffFile(input_tiff_path) as tif:
        current_slice = tif.pages[slice_index].asarray()

    # Compute patch means
    patch_means = []
    for i in range(0, current_slice.shape[0] - patch_x + 1, stride):
        for j in range(0, current_slice.shape[1] - patch_y + 1, stride):
            patch = current_slice[i:i+patch_x, j:j+patch_y]
            patch_means.append(np.mean(patch))

    # Compute Otsu threshold on patch means
    patch_means = np.array(patch_means)
    patch_mean_threshold = threshold_otsu(patch_means)

    # Threshold patches and reconstruct the image
    reconstructed_image = np.zeros_like(current_slice, dtype=float)
    patch_coords = []
    idx = 0
    for i in range(0, current_slice.shape[0] - patch_x + 1, stride):
        for j in range(0, current_slice.shape[1] - patch_y + 1, stride):
            if patch_means[idx] > patch_mean_threshold:
                thresholded_patch = np.ones((patch_x, patch_y))
                center_i = i + patch_x // 2
                center_j = j + patch_y // 2
                patch_coords.append((center_i, center_j))
            else:
                thresholded_patch = np.zeros((patch_x, patch_y))

            reconstructed_image[i:i+patch_x, j:j+patch_y] += (
                thresholded_patch * (reconstructed_image[i:i+patch_x, j:j+patch_y] == 0)
            )
            idx += 1

    reconstructed_image = (reconstructed_image) * 255

    # Save outputs
    output_image_path = os.path.join(output_dir, f"slice_{slice_index}.tiff")
    output_coords_path = os.path.join(output_dir, f"coords_{slice_index}.npy")
    tiff.imwrite(output_image_path, reconstructed_image.astype(np.uint8))
    np.save(output_coords_path, patch_coords)

    print(f"Processed slice {slice_index}, saved to {output_image_path}")

def main(args):
    """
    Main function to process the input file based on specified slice index and save outputs.

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    # python 02_get_binary_mask_per_slice.py /path/to/input.tiff /path/to/output_dir --slice_index 1 --patch_size 128 --stride 16
    """
    process_single_slice(args.input_file, args.slice_index, patch_x=args.patch_size, patch_y=args.patch_size, stride=args.stride, output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and segment single slice of 3D tissue data from a TIFF file")
    parser.add_argument('input_file', type=str, help="Path to input TIFF file")
    parser.add_argument('output_dir', type=str, help="Directory to save the processed slices")
    parser.add_argument('--slice_index', type=int, default=0, help="Index of the slice to process")
    parser.add_argument('--patch_size', type=int, default=128, help="Patch size for segmentation")
    parser.add_argument('--stride', type=int, default=16, help="Stride for patch overlap")

    args = parser.parse_args()
    main(args)
