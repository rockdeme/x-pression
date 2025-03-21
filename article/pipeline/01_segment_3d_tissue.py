import argparse
import numpy as np
import tifffile as tiff
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable


def split_image_into_patches_with_overlap(image, patch_size, stride):
    """
    Splits a 2D image into smaller patches with a given overlap (stride).

    Args:
        image (np.ndarray): The 2D image to be split.
        patch_size (int): The size of each patch == patch_size*patch_size.
        stride (int): The stride to control overlap between patches.

    Returns:
        tuple: A tuple containing the patches (np.ndarray) and their positions (list of tuples).
    """
    patches = []
    patch_positions = []
    height, width = image.shape

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            patch_positions.append((i, j))

    return np.array(patches), patch_positions

def stitch_patches_into_image(predictions, image_shape, patch_size, patch_positions):
    """
    Reconstructs a 2D image from its patches by averaging overlapping regions.

    Args:
        predictions (np.ndarray): The predicted patches to be stitched. Each patch is a 2D array of size `patch_size x patch_size`.
        image_shape (Tuple[int, int]): The shape (height, width) of the original image to be reconstructed.
        patch_size (int): The size of each square patch.
        patch_positions (List[Tuple[int, int]]): A list of tuples where each tuple (i, j) indicates the top-left position of a patch in the original image.

    Returns:
        np.ndarray: The reconstructed 2D image with averaged overlapping regions.
    """
    reconstructed_image = np.zeros(image_shape)
    count_map = np.zeros(image_shape)

    for idx, (i, j) in enumerate(patch_positions):
        reconstructed_image[i:i + patch_size, j:j + patch_size] += predictions[idx]
        count_map[i:i + patch_size, j:j + patch_size] += 1

    reconstructed_image /= count_map
    return reconstructed_image

@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient, which measures the similarity between two samples.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        smooth (float, optional): Smoothing constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        tf.Tensor: The Dice coefficient.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    """
    Computes the Dice loss, which is the complement of the Dice coefficient.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        tf.Tensor: The Dice loss.
    """
    return 1 - dice_coef(y_true, y_pred)

@register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def combine_slices(input_file, patch_size, stride, model, output_file):
    """
    Processes all slices from a 3D TIFF file, predicts each patch, and saves the resulting
    thresholded 2D predictions into a multi-page TIFF.

    Args:
        input_file (str): Path to input TIFF file.
        patch_size (int): Size of the patches.
        stride (int): Stride for patch overlap.
        model (keras.Model): Pre-trained model for segmentation.
        output_file (str): Path to save the output multi-page TIFF file.
    """
    with tiff.TiffFile(input_file) as tif, tiff.TiffWriter(output_file, bigtiff=True) as tif_writer:
        for _, page in enumerate(tif.pages):
            tomography_image = page.asarray()

            patches, patch_positions = split_image_into_patches_with_overlap(tomography_image, patch_size, stride)
            patches = patches / 255.0
            predictions = model.predict(patches, verbose=0)

            predictions = predictions.reshape(-1, patch_size, patch_size)

            stitched_image = stitch_patches_into_image(predictions, tomography_image.shape, patch_size, patch_positions)
            if np.any(np.isnan(stitched_image)):
                stitched_image = np.nan_to_num(stitched_image, nan=0.0)

            image = (stitched_image / np.max(stitched_image) * 255).astype(np.uint8)

            tif_writer.write(image, photometric='minisblack')

    print(f"Combined TIFF saved as: {output_file}")


def main(args):
    """
    Main function to load the model and process the input file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments
        python 01_segment_3d_tissue.py /path/to/input.tiff /path/to/unet_model.keras /path/to/output.tiff --patch_size 128 --stride 128 --slice_start 0

    """
    model = load_model(
        args.model_path,
        custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss, 'iou': iou}
    )
    combine_slices(args.input_file, args.patch_size, args.stride, model, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation of 3D tissue data using U-Net")
    parser.add_argument('input_file', type=str, help="Path to input 3D TIFF file")
    parser.add_argument('model_path', type=str, help="Path to the pre-trained U-Net model")
    parser.add_argument('output_file', type=str, help="Path to save the output multi-page TIFF file")
    parser.add_argument('--patch_size', type=int, default=128, help="Patch size for segmentation")
    parser.add_argument('--stride', type=int, default=128, help="Stride for patch overlap")

    args = parser.parse_args()
    main(args)
