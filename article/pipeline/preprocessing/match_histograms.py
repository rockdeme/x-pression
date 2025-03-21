import argparse
import numpy as np
import tifffile as tiff
from skimage import exposure
from skimage.transform import resize


def load_tiff_stack(file_path):
    """
    Loads a multi-page TIFF.
    """
    with tiff.TiffFile(file_path) as tif:
        volume = np.stack([page.asarray() for page in tif.pages])
    return volume


def normalize_image(image):
    """
    Normalizes an image to the range [0,1].
    """
    return (image - image.min()) / (image.max() - image.min())


def apply_histogram_matching(volume, reference_slice):
    """
    Applies histogram matching to each slice in the volume using a reference slice.

    Args:
        volume (np.ndarray): 3D array representing the image stack.
        reference_slice (np.ndarray): 2D reference slice for histogram matching.

    Returns:
        np.ndarray: 3D volume after histogram matching.
    """
    matched_volume = np.zeros_like(volume, dtype=np.uint8)

    for z in range(volume.shape[0]):
        slice_ = volume[z, :, :]

        if reference_slice.shape != slice_.shape:
            reference_slice = resize(reference_slice, slice_.shape, anti_aliasing=True)

        slice_norm = normalize_image(slice_)
        ref_norm = normalize_image(reference_slice)

        matched = exposure.match_histograms(slice_norm, ref_norm, channel_axis=None)
        # Convert to 8-bit and store in output volume
        matched_volume[z] = (matched * 255).clip(0, 255).astype(np.uint8)

        print(f"Processed slice {z + 1}/{volume.shape[0]}")

    return matched_volume


def save_tiff_stack(volume, output_path):
    """
    Saves a 3D NumPy array as a multi-page TIFF file.
    """
    with tiff.TiffWriter(output_path, bigtiff=True) as tif_writer:
        for slice_ in volume:
            tif_writer.write(slice_, contiguous=True)

    print(f"Histogram-matched TIFF saved as: {output_path}")


def main(args):
    """
    Main function to apply histogram matching to a 3D TIFF volume and a refference slice.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    reference_slice = load_tiff_stack(args.reference_file)[0]
    input_volume = load_tiff_stack(args.input_file)

    matched_volume = apply_histogram_matching(input_volume, reference_slice)
    save_tiff_stack(matched_volume, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply histogram matching to a 3D TIFF volume.")
    
    parser.add_argument("reference_file", type=str, help="Path to the reference TIFF slice.")
    parser.add_argument("input_file", type=str, help="Path to the input 3D TIFF volume.")
    parser.add_argument("output_file", type=str, help="Path to save the histogram-matched TIFF volume.")

    args = parser.parse_args()
    main(args)
