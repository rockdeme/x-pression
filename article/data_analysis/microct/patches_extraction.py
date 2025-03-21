import argparse
import scanpy as sc
import pandas as pd
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

def load_data(adata_path, compartments_path):
    """Loads the h5ad file and compartments CSV, merging them."""
    adata = sc.read_h5ad(adata_path)
    df = pd.read_csv(compartments_path)
    df.rename(columns={"Unnamed: 0": "cell_id"}, inplace=True)
    df.set_index("cell_id", inplace=True)
    adata.obs = adata.obs.merge(df, left_index=True, right_index=True, how="left")
    return adata


def filter_cells_by_class(adata, target_class, threshold):
    """Filters cells by selected class and probability threshold."""
    obs_columns = ['0', '1', '2', '3', '4', '5', '6', '7']
    df_obs = adata.obs[obs_columns].copy()
    df_spatial_transformed = pd.DataFrame(
        adata.obsm['spatial_transformed'], index=adata.obs.index, columns=['X', 'Y']
    )
    df_final = pd.concat([df_obs, df_spatial_transformed], axis=1)

    df_final['selected_class'] = df_final[obs_columns].idxmax(axis=1)
    selected_class_df = df_final[df_final['selected_class'] == str(target_class)]
    df_filtered = selected_class_df[(selected_class_df[obs_columns] > threshold).any(axis=1)]
    
    return df_filtered

def extract_patches(input_tiff, df_filtered, output_dir, z_slice, target_class):
    """Extracts patches from the TIFF image based on the filtered data."""
    z_crop_s, z_crop_e = z_slice - 10, z_slice + 11

    with tifffile.TiffFile(input_tiff) as tif:
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing patches"):
            x_center, y_center = int(row["X"]), int(row["Y"])
            y_crop_s, y_crop_e = y_center - 64, y_center + 64
            x_crop_s, x_crop_e = x_center - 64, x_center + 64

            output_png = f"{output_dir}/class{target_class}_patch_{x_center}_{y_center}.png"

            cropped_slices = [
                tif.pages[z].asarray()[y_crop_s:y_crop_e, x_crop_s:x_crop_e] 
                for z in range(z_crop_s, z_crop_e)
            ]
            volume = np.array(cropped_slices)

            sagittal_img = volume[20:22, :128, :128].squeeze(0)
            axial_img = volume[:22, 127:128, :128].squeeze(1)
            coronal_img = volume[:22, :128, 127:128].squeeze(2)

            plot_3d_patch(sagittal_img, axial_img, coronal_img, output_png)


def plot_3d_patch(sagittal_img, axial_img, coronal_img, output_png):
    """Plots and saves the 3D patch visualization."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.dist = 6.2
    ax.view_init(elev=38, azim=225)

    norm_ax = Normalize(vmin=np.percentile(axial_img, 1), vmax=np.percentile(axial_img, 95))
    norm_cor = Normalize(vmin=np.percentile(coronal_img, 1), vmax=np.percentile(coronal_img, 95))
    norm_sag = Normalize(vmin=np.percentile(sagittal_img, 1), vmax=np.percentile(sagittal_img, 95))

    axial_colored = cm.gray(norm_ax(axial_img))
    coronal_colored = cm.gray(norm_cor(coronal_img))
    sagittal_colored = cm.gray(norm_sag(sagittal_img))

    sagittal_colored = np.flipud(np.fliplr(sagittal_colored))
    coronal_colored = np.fliplr(coronal_colored)
    axial_colored = np.fliplr(axial_colored)

    xp_ax, yp_ax = axial_img.shape
    xp_cor, yp_cor = coronal_img.shape
    xp_sag, yp_sag = sagittal_img.shape

    Y_ax, X_ax = np.meshgrid(np.arange(yp_ax), np.arange(xp_ax))
    Y_cor, X_cor = np.meshgrid(np.arange(yp_cor), np.arange(xp_cor))
    Y_sag, X_sag = np.meshgrid(np.arange(yp_sag), np.arange(xp_sag))

    ax.plot_surface(X_sag, Y_sag, np.full_like(X_sag, 20), facecolors=sagittal_colored, shade=False)
    ax.plot_surface(Y_cor, np.full_like(Y_cor, 0), X_cor, facecolors=coronal_colored, shade=False)
    ax.plot_surface(np.full_like(X_ax, 0), Y_ax, X_ax, facecolors=axial_colored, shade=False)

    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])
    ax.set_zlim([0, 128])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    plt.savefig(output_png, format='png', dpi=300, transparent=True)
    plt.close(fig)


def main(args):
    """Main function to extract patches.
        # python [patches_extraction].py \
        --adata "/path/to/adata.h5ad" \
        --compartments "/path/to/compartments.csv" \
        --input_tiff "/path/to/image.tiff" \
        --output_dir "/path/to/output/" \
        --z_slice 296 \
        --patch_size 128 \
        --target_class 7 \
        --threshold 0.7
    """
    adata = load_data(args.adata, args.compartments)
    df_filtered = filter_cells_by_class(adata, args.target_class, args.threshold)
    extract_patches(args.input_tiff, df_filtered, args.output_dir, args.z_slice, args.target_class)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 3D patches from TIFF based on spatial data.")
    parser.add_argument('--adata', type=str, required=True, help="Path to the h5ad file.")
    parser.add_argument('--compartments', type=str, required=True, help="Path to the compartments CSV file.")
    parser.add_argument('--input_tiff', type=str, required=True, help="Path to the input TIFF file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images.")
    parser.add_argument('--z_slice', type=int, default=296, help="Central Z slice for cropping.")
    parser.add_argument('--patch_size', type=int, default=128, help="Size of the extracted patch.")
    parser.add_argument('--target_class', type=str, required=True, help="Class label to filter (e.g., '7').")
    parser.add_argument('--threshold', type=float, default=0.7, help="Probability threshold for selecting class.")

    args = parser.parse_args()
    main(args)
