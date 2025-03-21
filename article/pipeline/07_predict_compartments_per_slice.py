import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile as tiff


def predict_3d_per_slice(slice_number, output_dir, input_tiff_path, patch_coords_path, model_path, patch_size=(128, 128, 21), batch_size=500):
    """
    Predict the class for each patch of a specific slice and save the predictions in a CSV file.

    Parameters:
        slice_number (int): The slice number to process.
        output_dir (str): Directory to save the output CSV file.
        input_tiff_path (str): Path to the input TIFF file containing CT data.
        patch_coords_path (str): Path to the file containing patch coordinates for each slice.
        model_path (str): Path to the trained model.
        patch_size (tuple): The size of the patches (default: (128, 128, 21)).
        batch_size (int): Batch size for processing (default: 500).
    """
    patch_coords_per_slice = np.load(patch_coords_path, allow_pickle=True)
    model = tf.keras.models.load_model(model_path)

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"predictions_slice_{slice_number}.csv")

    half_height = patch_size[0] // 2
    half_width = patch_size[1] // 2
    offset = patch_size[2] // 2

    with tiff.TiffFile(input_tiff_path) as tif:
        num_slices = len(tif.pages)
        
        # Skip invalid slices
        if slice_number - offset < 0 or slice_number + offset >= num_slices:
            print(f"Skipping slice {slice_number}: Out of valid range.")
            return
        
        slice_coords = patch_coords_per_slice[slice_number - offset]

        slice_start = slice_number - offset
        slice_end = slice_number + offset + 1
        volume_patches = np.stack([tif.pages[s].asarray() for s in range(slice_start, slice_end)])

        all_predictions = []
        num_batches = (len(slice_coords) + batch_size - 1) // batch_size

        # Predictions per batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(slice_coords))
            batch_coords = slice_coords[start_idx:end_idx]

            patches = []
            for (center_y, center_x) in batch_coords:
                y_start = center_y - half_height
                x_start = center_x - half_width

                if (
                    y_start >= 0 and x_start >= 0 and
                    y_start + patch_size[0] <= volume_patches.shape[1] and
                    x_start + patch_size[1] <= volume_patches.shape[2]
                ):
                    volume_patch = volume_patches[
                        :, y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]
                    ]
                    volume_patch = volume_patch.astype(float) / 255.0
                    volume_patch = np.expand_dims(volume_patch, axis=-1)  # Add channel dim
                    patches.append(volume_patch)

            if patches:
                patches = np.stack(patches)
                batch_predictions = model.predict(patches)

                for patch_idx, prediction in enumerate(batch_predictions):
                    all_predictions.append({
                        'slice_number': slice_number,
                        'patch_index': start_idx + patch_idx,
                        **{f'class_{i}': pred for i, pred in enumerate(prediction)}
                    })

        # Save predictions to CSV
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved for slice {slice_number}: {output_csv_path}")


def main(args):
    """
    Main function to handle argument parsing and call the prediction function.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    predict_3d_per_slice(
        args.slice_number,
        args.output_dir,
        args.input_tiff,
        args.patch_coords,
        args.model_path,
        patch_size=args.patch_size,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict patches for a given slice of 3D data and save the predictions.")
    parser.add_argument('slice_number', type=int, help="The slice number to process.")
    parser.add_argument('output_dir', type=str, help="Directory to save the output CSV file.")
    parser.add_argument('input_tiff', type=str, help="Path to the input TIFF file containing CT data.")
    parser.add_argument('patch_coords', type=str, help="Path to the file containing patch coordinates for each slice.")
    parser.add_argument('model_path', type=str, help="Path to the trained model.")
    parser.add_argument('--patch_size', type=tuple, default=(128, 128, 21), help="Patch size (default: (128, 128, 21)).")
    parser.add_argument('--batch_size', type=int, default=500, help="Batch size for processing patches (default: 500).")

    args = parser.parse_args()
    main(args)
