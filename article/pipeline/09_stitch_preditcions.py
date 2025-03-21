import numpy as np
import pandas as pd
import tifffile as tiff
import os
import argparse

def process_tiff_and_predictions(input_tiff_path, patch_coords_path, compartment_predictions_csv, output_file, patch_size=(16, 16, 21)):
    """
    Process the input TIFF file, patch coordinates, and PCA predictions to generate a combined output TIFF.

    Args:
        input_tiff_path (str): Path to the input TIFF file.
        patch_coords_path (str): Path to the patch coordinates file.
        compartment_predictions_csv (str): Path to the PCA predictions CSV file.
        output_file (str): Path to save the combined output TIFF.
        patch_size (tuple): Size of the patches (default is (16, 16, 21)).
    """
    # Check if input files exist
    for file in [input_tiff_path, patch_coords_path, compartment_predictions_csv]:
        if not os.path.exists(file):
            print(f"Error: File not found - {file}")
            exit()

    # Patch size and offsets
    half_height = patch_size[0] // 2
    half_width = patch_size[1] // 2
    offset = patch_size[2] // 2  # Offset to align slices with predictions

    # Load patch coordinates and PCA predictions
    patch_coords_per_slice = np.load(patch_coords_path, allow_pickle=True)
    pca_predictions_df = pd.read_csv(compartment_predictions_csv)

    print(f"Loaded patch coordinates: {patch_coords_per_slice.shape}")
    print(f"Loaded PCA predictions: {pca_predictions_df.shape}")

    # Open TIFF and prepare output TIFF writer
    with tiff.TiffFile(input_tiff_path) as tif, tiff.TiffWriter(output_file, bigtiff=True) as t:
        num_slices = len(tif.pages)
        print(f"Total slices in TIFF: {num_slices}")

        # Loop over slices (adjust range if necessary)
        for slice_number in range(21, num_slices, 21):
            # Check if predictions exist for this slice
            slice_predictions = pca_predictions_df[pca_predictions_df['slice_number'] == slice_number]
            if slice_predictions.empty:
                print(f"Skipping slice {slice_number}: No predictions found.")
                continue

            # Get original slice and prepare storage for stitched predictions
            original_slice = tif.pages[slice_number].asarray().astype(np.float16)
            print(f"Original slice shape: {original_slice.shape}, max value: {np.max(original_slice)}")
            stitched_images = [np.zeros(original_slice.shape, dtype=np.float16) for _ in range(8)]

            # Get patch coordinates for this slice
            slice_coords = patch_coords_per_slice[slice_number - offset]
            print(f"Number of patches in slice {slice_number}: {len(slice_coords)}")
            print(f"Number of predictions in slice {slice_number}: {len(slice_predictions)}")

            # Stitch predictions into the image
            for patch_index, (center_y, center_x) in enumerate(slice_coords[:len(slice_predictions)]):
                y_start = center_y - half_height
                x_start = center_x - half_width

                if (
                    y_start >= 0 and x_start >= 0 and
                    y_start + patch_size[0] <= original_slice.shape[0] and
                    x_start + patch_size[1] <= original_slice.shape[1]
                ):
                    patch_probs = slice_predictions.iloc[patch_index, 2:].values  # Extract class probabilities
                    for class_idx in range(8):
                        stitched_images[class_idx][y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]] = patch_probs[class_idx]

            # Prepare final multidimensional image (normalized)
            MultiDimImg = np.zeros((original_slice.shape[0], original_slice.shape[1], 9), dtype=np.float16)

            # Normalize original slice to [0, 1]
            MultiDimImg[:, :, 0] = original_slice / np.max(original_slice) if np.max(original_slice) > 0 else 0

            # Add class probabilities
            for class_idx in range(8):
                MultiDimImg[:, :, class_idx + 1] = stitched_images[class_idx]

            print(f"Final MultiDimImg shape: {MultiDimImg.shape}")
            print(f"Max pixel value in MultiDimImg: {np.max(MultiDimImg)}")

            # Write each channel to TIFF
            for channel_idx in range(MultiDimImg.shape[2]):
                t.write(MultiDimImg[:, :, channel_idx], dtype=np.float16, photometric='minisblack')

    print(f"Combined TIFF saved as: {output_file}")

def main(args):
    """
    Main function to process patch coordinates for the slices.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        # python 09_stitch_predictions.py /path/to/input_tiff.tiff /path/to/patch_coords.npy /path/to/pca_predictions.csv /path/to/output_file.tiff --patch_size 128 128 21
    """
    process_tiff_and_predictions(args.input_tiff, args.patch_coords, args.compartment_predictions_csv, args.output_file, patch_size=args.patch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patch coordinates for each slice and combine predictions into a TIFF.")
    parser.add_argument('input_tiff', type=str, help="Path to input TIFF file (from 05_extract_masked_ct.py script)")
    parser.add_argument('patch_coords', type=str, help="Path to file containing patch coordinates")
    parser.add_argument('compartment_predictions_csv', type=str, help="Path to compartment predictions CSV file")
    parser.add_argument('output_file', type=str, help="Path to save the combined output TIFF")
    parser.add_argument('--patch_size', type=int, nargs=3, default=(16, 16, 21), help="Size of patches (default: 16x16x21)")

    args = parser.parse_args()
    main(args)
