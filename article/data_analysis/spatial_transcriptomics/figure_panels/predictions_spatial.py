import pandas as pd
import scanpy as sc
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


file_dir = '/mnt/f/3d_tomography/cnn_training/output/results_v3/'
experiment = 'single_sample_v3_75epoch_64-64-21'

labels_df = pd.DataFrame()
csvs = glob(file_dir + experiment + '_predictions*.csv')
for f in csvs:
    split = f.split('_')[-1][:-4].lower()
    df = pd.read_csv(f, index_col=0)
    if split == 'val':
        df['split'] = 'validation'
    else:
        df['split'] = split
    labels_df = pd.concat([labels_df, df], ignore_index=True)

labels_df = labels_df.drop_duplicates(subset=['barcode'])
labels_df.index = labels_df['barcode']

print(labels_df['split'].value_counts())

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')

compartments_df = pd.DataFrame(adata.obsm['chr_aa'], index=adata.obs.index)
adata = adata[adata.obs.index.isin(list(labels_df['barcode']))]

for i in range(compartments_df.shape[1]):
    adata.obs[f'{i}_y'] = compartments_df[i]
    adata.obs[f'{i}_y_pred'] = labels_df[str(i)]

n_cols = 4
n_rows = 4
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
sc.set_figure_params(vector_friendly=True, dpi_save=100)
for i in range(4):
    sc.pl.spatial(adata, color=f'{i}_y', s=12, ax=axes[i, 0], show=False, colorbar_loc=None,
                  vmin=0, vmax=1, alpha_img=0.7)
    axes[i, 0].set_title(f'True values\nCompartment {i}')

    sc.pl.spatial(adata, color=f'{i}_y_pred', s=12, ax=axes[i, 1], show=False, colorbar_loc=None,
                  vmin=0, vmax=1, alpha_img=0.7)
    axes[i, 1].set_title(f'Predictions\nCompartment {i}')

    sc.pl.spatial(adata, color=f'{i + 4}_y', s=12, ax=axes[i, 2], show=False, colorbar_loc=None,
                  vmin=0, vmax=1, alpha_img=0.7)
    axes[i, 2].set_title(f'True values\nCompartment {i + 4}')

    sc.pl.spatial(adata, color=f'{i + 4}_y_pred', s=12, ax=axes[i, 3], show=False, colorbar_loc=None,
                  vmin=0, vmax=1, alpha_img=0.7)
    axes[i, 3].set_title(f'Predictions\nCompartment {i + 4}')

for i in range(4):
    for j in range(4):
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')
        axes[i, j].spines['top'].set_visible(False)
        axes[i, j].spines['right'].set_visible(False)
        axes[i, j].spines['left'].set_visible(False)
        axes[i, j].spines['bottom'].set_visible(False)

    divider = make_axes_locatable(axes[i, 3])
    cax = divider.append_axes("right", size="5%", pad="3%")
    sc_img = axes[i, 3].collections[0]
    colorbar = plt.colorbar(sc_img, cax=cax, format="%.1f")
    colorbar.ax.set_aspect(10)
    cax.set_ylabel('Score')

# Adjust layout for spacing
plt.tight_layout()
plt.savefig('data/figures/training_true_pred_spatial.svg')
plt.show()
