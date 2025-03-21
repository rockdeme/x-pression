import math
import argparse
import SimpleITK as sitk
import numpy as np

def read_image(file_path):
    """
    Reads a TIFF image.
    """
    return sitk.ReadImage(file_path)

def apply_transformation_and_save(input_image, row_angle, col_angle, output_path):
    """
    Applies a rotation transformation to the input image and saves the result.

    Args:
        input_image (SimpleITK.Image): The image to be transformed.
        row_angle (float): The rotation angle around the x-axis in degrees.
        col_angle (float): The rotation angle around the y-axis in degrees.
        output_path (str): Path to save the transformed image.
    """
    transform = sitk.Euler3DTransform()
    transform.SetRotation(
        row_angle * math.pi / 180.0,
        -col_angle * math.pi / 180.0,
        0.0
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(input_image.GetSize())
    resampler.SetOutputSpacing(input_image.GetSpacing())
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetDefaultPixelValue(input_image.GetPixelIDValue())

    output_image = resampler.Execute(input_image)
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.SetImageIO("TIFFImageIO")
    writer.Execute(output_image)

def main(args):
    """
    Main function to processes the input TIFF file, applies the transformation, and saves the aligned image.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        # python rotate.py /path/to/input.tiff /path/to/output.tiff --row_angle -1 --col_angle -3 

    """
    input_image = read_image(args.file_path)

    # Apply the rotation transformation and save the output
    apply_transformation_and_save(input_image, args.row_angle, args.col_angle, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and align a TIFF image volume.")
    parser.add_argument('file_path', type=str, help="Path to the input TIFF file.")
    parser.add_argument('output_path', type=str, help="Path to save the aligned TIFF volume.")
    
    # rotation angles were found with 3DSlicer
    parser.add_argument('row_angle', type=float, help="Rotation angle around the x-axis (degrees).")
    parser.add_argument('col_angle', type=float, help="Rotation angle around the y-axis (degrees).")

    args = parser.parse_args()
    main(args)
