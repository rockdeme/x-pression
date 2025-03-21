import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch
import matplotlib.pyplot as plt


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(f'{sample_folder}/dataset.h5ad')

ch.harmony_integration(adata, 'chr_sample_id', random_state=42, block_size=0.05)

adata.write(f'{sample_folder}/dataset_harmony.h5ad')

ch.plot_explained_variance(adata)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(x=adata.obsm['chr_X_pca'][:, 3], y=adata.obsm['chr_X_pca'][:, 4],
           rasterized=True, s=4, c=adata.obs['chr_sample_id'].cat.codes, cmap='tab20', alpha=0.1)
plt.tight_layout()
plt.show()
