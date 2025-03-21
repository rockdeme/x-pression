import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch
import matplotlib.pyplot as plt


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(f'{sample_folder}/dataset.h5ad')
# adata = adata[:, adata.var['spatially_variable'] == True]  # to fit this into memory without crashing

ch.pca(adata, n_pcs=50)

adata.write(f'{sample_folder}/dataset.h5ad')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(x=adata.obsm['chr_X_pca'][:, 0], y=adata.obsm['chr_X_pca'][:, 1],
           rasterized=True, s=4, c=adata.obs['chr_sample_id'].cat.codes, cmap='tab20', alpha=0.5)
plt.tight_layout()
plt.show()
