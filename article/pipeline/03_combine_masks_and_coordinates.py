import argparse
import os
import re
from glob import glob
import numpy as np
import tifffile as tiff

def numerical_sort_slice(filename):
    """
    Extracts the numerical slice index from the filename for sorting.
    """
    match = re.search(r'slice_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def numerical_sort_coord(filename):
    """
    Extracts the numerical coordinate index from the filename for sorting.
    """
    match = re.search(r'coords_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def combine_results(input_dir, combined_tiff_path, combined_coords_path):
    """
    Combine processed slices and patch coordinates into single files.

    Parameters:
        input_dir (str): Directory containing slice results.
        combined_tiff_path (str): Path for the combined TIFF file.
        combined_coords_path (str): Path for the combined coordinates file.
    """
    slice_files = sorted(glob(os.path.join(input_dir, "slice_*.tiff")), key=numerical_sort_slice)
    coord_files = sorted(glob(os.path.join(input_dir, "coords_*.npy")), key=numerical_sort_coord)

    combined_coords = []
    with tiff.TiffWriter(combined_tiff_path, bigtiff=True) as tiff_writer:
        for slice_file in slice_files:
            slice_data = tiff.imread(slice_file)
            tiff_writer.write(slice_data.astype(np.uint8))
        
        for coord_file in coord_files:
            coords = np.load(coord_file, allow_pickle=True)
            combined_coords.append(coords)
    
    np.save(combined_coords_path, np.array(combined_coords, dtype=object))

    print(f"Combined TIFF saved to: {combined_tiff_path}")
    print(f"Combined coordinates saved to: {combined_coords_path}")

def main(args):
    """
    Main function to combine the processed results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments
        # python 03_combine_masks_and_coordinates.py path/to/input_dir path/to/combined_tiff path/to/combined_coords
    """
    combine_results(args.input_dir, args.combined_tiff_path, args.combined_coords_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine processed slices and coordinates into single files")
    parser.add_argument('input_dir', type=str, help="Directory containing the processed slice files")
    parser.add_argument('combined_tiff_path', type=str, help="Path to save the combined TIFF file")
    parser.add_argument('combined_coords_path', type=str, help="Path to save the combined coordinates file")

    args = parser.parse_args()
    main(args)
