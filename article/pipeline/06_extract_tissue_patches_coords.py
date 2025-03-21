import argparse
import numpy as np
import tifffile as tiff
from tqdm import tqdm

def process_patch_coords(input_tiff_path, patch_coords_file, patch_coords_path, patch_size=(128, 128, 21)):
    """
    Process the patch coordinates for each slice and save the valid coordinates for patches that are tissue.
    
    Parameters:
        input_tiff_path (str): Path to the input TIFF file.
        patch_coords_file (str): Path to the file containing patch coordinates for all slices.
        patch_coords_path (str): Path to save the valid patch coordinates.
        patch_size (tuple): The size of the patches (default: 128x128x21).
    """
    depth = patch_size[2]
    half_depth = depth // 2
    half_height = patch_size[0] // 2
    half_width = patch_size[1] // 2

    # Load patch coordinates
    patch_coords_all_slices = np.load(patch_coords_file, allow_pickle=True)
    print(f"Loaded {len(patch_coords_all_slices)} slices with patch coordinates.")

    valid_patch_coords = [[] for _ in range(len(patch_coords_all_slices))]

    with tiff.TiffFile(input_tiff_path) as tif:
        num_pages = len(tif.pages)
        print(f"The TIFF file contains {num_pages} slices.")

        for center_idx in tqdm(range(half_depth, num_pages - half_depth), desc="Processing slices"):
            patch_slices = np.stack([
                tif.pages[z].asarray() for z in range(center_idx - half_depth, center_idx + half_depth + 1)
            ], axis=0)  # Shape: (depth, height, width)

            patch_coords = patch_coords_all_slices[center_idx]

            for (center_y, center_x) in patch_coords:
                y_start = center_y - half_height
                x_start = center_x - half_width
                patch = patch_slices[:, y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]

                # Check if the patch contains all 1s (255 in the mask)
                if np.all(patch == 255):
                    valid_patch_coords[center_idx].append((center_y, center_x))

    np.save(patch_coords_path, np.array(valid_patch_coords, dtype=object))
    print(f"Saved {sum(len(coords) for coords in valid_patch_coords)} valid patch coordinates to {patch_coords_path}.")


def main(args):
    """
    Main function to process patch coordinates for the slices.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        # python extract_tissue_patches_coords.py /path/to/input.tiff /path/to/patch_coords.npy /path/to/output_coords.npy --patch_size 128 128 21

    """
    process_patch_coords(args.input_file, args.patch_coords_file, args.output_file, patch_size=args.patch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patch coordinates for each slice and save the valid coordinates.")
    parser.add_argument('input_file', type=str, help="Path to input TIFF file")
    parser.add_argument('patch_coords_file', type=str, help="Path to file containing patch coordinates")
    parser.add_argument('output_file', type=str, help="Path to save the valid patch coordinates")
    parser.add_argument('--patch_size', type=int, nargs=3, default=(128, 128, 21), help="Size of patches (default: 128x128x21)")

    args = parser.parse_args()
    main(args)
