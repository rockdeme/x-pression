import argparse
import numpy as np
import pandas as pd
import scanpy as sc

def load_landmarks(file_path):
    """
    Load landmark points from CSV (without header).
    Args:
        file_path (str): Path to the landmarks CSV file.
    Returns:
        np.ndarray, np.ndarray: Fixed and moving points as NumPy arrays.
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ["Point", "Used", "Fixed_X", "Fixed_Y", "Moving_X", "Moving_Y"]
    fixed_points = df[["Fixed_X", "Fixed_Y"]].to_numpy()
    moving_points = df[["Moving_X", "Moving_Y"]].to_numpy()
    return fixed_points, moving_points

def compute_affine_matrix(fixed_points, moving_points):
    """
    Compute the affine transformation matrix using least squares.
    Args:
        fixed_points (np.ndarray): Fixed reference points (Nx2).
        moving_points (np.ndarray): Corresponding moving points (Nx2).
    Returns:
        np.ndarray: 3x3 affine transformation matrix.
    """
    A, B = [], []
    for (x, y), (x_prime, y_prime) in zip(fixed_points, moving_points):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(x_prime)
        B.append(y_prime)

    A, B = np.array(A), np.array(B)
    params = np.linalg.lstsq(A, B, rcond=None)[0]

    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]],
                     [0, 0, 1]])

def apply_affine_transformation(coords, matrix):
    """
    Apply affine transformation to a set of coordinates.
    Args:
        coords (np.ndarray): Original spatial coordinates (Nx2).
        matrix (np.ndarray): 3x3 affine transformation matrix.
    Returns:
        np.ndarray: Transformed coordinates (Nx2).
    """
    homogeneous_coords = np.hstack([coords, np.ones((coords.shape[0], 1))])
    transformed_coords_homogeneous = (matrix @ homogeneous_coords.T).T
    return transformed_coords_homogeneous[:, :2] / transformed_coords_homogeneous[:, 2, np.newaxis]

def main(args):
    """
    Main function to apply an affine transformation to spatial coordinates.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    adata = sc.read_h5ad(args.input_h5ad)
    spatial_coords = adata.obsm["spatial"]  # Extract original spatial coordinates

    fixed_points, moving_points = load_landmarks(args.landmarks_csv)

    affine_matrix = compute_affine_matrix(fixed_points, moving_points)
    transformed_coords = apply_affine_transformation(spatial_coords, affine_matrix)

    adata.obsm["transformed_spatial"] = transformed_coords
    adata.write(args.output_h5ad)

    print(f"Affine transformation complete. Transformed coordinates saved to {args.output_h5ad}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply affine transformation to spatial coordinates in AnnData.")
    
    parser.add_argument("input_h5ad", type=str, help="Path to the input .h5ad file.")
    parser.add_argument("landmarks_csv", type=str, help="Path to the landmarks CSV file (no header).")
    parser.add_argument("output_h5ad", type=str, help="Path to save the transformed .h5ad file.")

    args = parser.parse_args()
    main(args)
