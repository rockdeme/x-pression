import argparse
import numpy as np
import tifffile as tiff

def extract_masked_ct(ct_path, mask_path, output_path, num_slices_to_process=None):
    """
    Extract CT slices masked by the binary mask and save them to a new TIFF file.
    
    Parameters:
        ct_path (str): Path to the input CT TIFF file.
        mask_path (str): Path to the input binary mask TIFF file.
        output_path (str): Path to save the extracted and masked CT slices.
        num_slices_to_process (int or None): Number of slices to process. If None, process all slices.
    """
    with tiff.TiffWriter(output_path, bigtiff=True) as tif_writer:
        with tiff.TiffFile(ct_path) as ct_tif, tiff.TiffFile(mask_path) as mask_tif:
            if num_slices_to_process is None:
                num_slices_to_process = len(ct_tif.pages)

            for i in range(min(num_slices_to_process, len(ct_tif.pages))):
                ct_slice = ct_tif.pages[i].asarray()
                mask_slice = mask_tif.pages[i].asarray()

                if ct_slice.shape != mask_slice.shape:
                    raise ValueError(f"Shape mismatch at slice {i}: CT {ct_slice.shape}, Mask {mask_slice.shape}")

                masked_slice = np.where(mask_slice == 255, ct_slice, 0)
                tif_writer.write(masked_slice, dtype=masked_slice.dtype)

    print(f"Extracted CT regions for the first {num_slices_to_process} slices saved to {output_path}.")

def main(args):
    """
    Main function to extract CT slices masked by the binary mask.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        # python extract_masked_ct.py /path/to/ct.tiff /path/to/mask.tiff /path/to/output.tiff --num_slices 10
        # python extract_masked_ct.py /path/to/ct.tiff /path/to/mask.tiff /path/to/output.tiff
    """
    extract_masked_ct(args.ct_file, args.mask_file, args.output_file, num_slices_to_process=args.num_slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CT slices masked by a binary mask")
    parser.add_argument('ct_file', type=str, help="Path to input CT TIFF file")
    parser.add_argument('mask_file', type=str, help="Path to input binary mask TIFF file")
    parser.add_argument('output_file', type=str, help="Path to save the extracted CT masked slices")
    parser.add_argument('--num_slices', type=int, default=None, help="Number of slices to process. If not provided, all slices will be processed.")

    args = parser.parse_args()
    main(args)
