import argparse
import numpy as np
import tifffile as tiff
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def remove_small_objects(input_tiff_path, output_tiff_path, patch_size=(128, 128), min_objects=2):
    """
    Remove small objects from the mask that are smaller than the specified number of patches.
    
    Parameters:
        input_tiff_path (str): Path to the input TIFF file containing the binary mask.
        output_tiff_path (str): Path to save the cleaned mask.
        patch_size (tuple): The size of the patches (default 128x128).
        min_objects (int): Minimum number of patches an object must have to be retained (default is 2, we recommend 5).
    """
    patch_area = patch_size[0] * patch_size[1]
    
    with tiff.TiffFile(input_tiff_path) as tif:
        processed_slices = []

        for slice_index, current_slice in enumerate(tif.pages):  

            binary_mask = current_slice.asarray()
            binary_mask = binary_mask // 255

            labeled_mask = label(binary_mask)
            regions = regionprops(labeled_mask)
            cleaned_mask = np.zeros_like(binary_mask)

            for region in regions:
                region_area = region.area

                # If the area is large enough (i.e., the region contains at least `min_objects` patches)
                if region_area >= min_objects * patch_area:
                    cleaned_mask[labeled_mask == region.label] = 1  # Keep this object

            # Convert the cleaned mask back to 255 (as a binary mask)
            processed_slice = cleaned_mask.astype(np.uint8) * 255
            processed_slices.append(processed_slice)
            print(f"Finished slice {slice_index + 1}/{len(tif.pages)}")

        with tiff.TiffWriter(output_tiff_path, bigtiff=True) as tiff_writer:
            print('writing')
            for img in processed_slices:
                print('per img')
                tiff_writer.write(img)

    print(f"Processed mask saved to {output_tiff_path}")

def main(args):
    """
    Main function to remove small objects from a binary mask.

    Args:
        args (argparse.Namespace): The parsed command-line arguments
        # python clean_binary_mask.py /path/to/input.tiff /path/to/output.tiff --patch_size 128 --min_objects 5
    """
    remove_small_objects(args.input_file, args.output_file, patch_size=(args.patch_size, args.patch_size), min_objects=args.min_objects)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove small objects from a binary mask")
    parser.add_argument('input_file', type=str, help="Path to input binary mask TIFF file")
    parser.add_argument('output_file', type=str, help="Path to save the cleaned binary mask TIFF file")
    parser.add_argument('--patch_size', type=int, default=128, help="Size of patches to define object area (default: 128)")
    parser.add_argument('--min_objects', type=int, default=2, help="Minimum number of patches an object must have to be retained (default: 2)")

    args = parser.parse_args()
    main(args)