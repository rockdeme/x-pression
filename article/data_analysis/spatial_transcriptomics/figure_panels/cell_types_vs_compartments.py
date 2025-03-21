import numpy as np
import pandas as pd
import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from data_analysis.utils import matrixplot

#%%

# EXP 1
# read adata
sample_path = 'data/spatial_transcriptomics/cell_type_deconvolution/exp1/cell2location_map/sp_1.h5ad'
adata = sc.read_h5ad(sample_path)

# create df from chrysalis array
comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)
# adata.obsm['chr_aa'] = comps_df
# add compartment colors
hexcodes = ch.utils.get_hexcodes(None, 8, 42, len(adata))

adata.obs['label_exp1'] = adata.obs['condition'].astype(str) + '-' + adata.obs['timepoint'].astype(str)
# adata.obs['label_exp2'] = (adata.obs['condition'].astype(str) +
#                           '-' + adata.obs['timepoint'].astype(str) +
#                           '-' + adata.obs['challenge'].astype(str))

#%%
adata.obsm['cell2loc']  = pd.DataFrame(adata.obsm['q05_cell_abundance_w_sf'].values,
                                       columns=adata.uns['mod']['factor_names'], index=adata.obs.index)
celltypes_df = adata.obsm['cell2loc']

corr_matrix = np.empty((len(celltypes_df.columns), len(comps_df.columns)))
for i, col1 in enumerate(celltypes_df.columns):
    for j, col2 in enumerate(comps_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], comps_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=celltypes_df.columns,
                     columns=comps_df.columns).T

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

matrixplot(corrs, figsize=(5.55, 5), flip=True, scaling=False, square=True,
            colorbar_shrink=0.325, colorbar_aspect=10, title='Cell type contributions\nto tissue compartments',
            dendrogram_ratio=0.05, cbar_label="Pearsons's r", xlabel='Cell types',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Tissue compartment', rasterized=True, seed=42, linewidths=0.0, xrot=45,
            reorder_obs=True)
plt.savefig(f'data/figures/cell_types_vs_comps.svg')
plt.show()

#%%

# EXP 2
# read adata
sample_path_1 = 'data/spatial_transcriptomics/cell_type_deconvolution/exp2/cell2location_map/sp_2_1.h5ad'
sample_path_2 = 'data/spatial_transcriptomics/cell_type_deconvolution/exp2/cell2location_map/sp_2_2.h5ad'

ad1 = sc.read_h5ad(sample_path_1)
ad2 = sc.read_h5ad(sample_path_2)

ad1.obsm['cell2loc']  = pd.DataFrame(ad1.obsm['q05_cell_abundance_w_sf'].values,
                                       columns=ad1.uns['mod']['factor_names'], index=ad1.obs.index)
ad2.obsm['cell2loc']  = pd.DataFrame(ad2.obsm['q05_cell_abundance_w_sf'].values,
                                       columns=ad2.uns['mod']['factor_names'], index=ad2.obs.index)
adata = sc.AnnData.concatenate(ad1, ad2, join="outer", index_unique=None)

# create df from chrysalis array
comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)
# adata.obsm['chr_aa'] = comps_df
# add compartment colors
hexcodes = ch.utils.get_hexcodes(None, 8, 42, len(adata))

adata.obs['label_exp2'] = (adata.obs['condition'].astype(str) +
                          '-' + adata.obs['timepoint'].astype(str) +
                          '-' + adata.obs['challenge'].astype(str))

#%%

celltypes_df = adata.obsm['cell2loc']

corr_matrix = np.empty((len(celltypes_df.columns), len(comps_df.columns)))
for i, col1 in enumerate(celltypes_df.columns):
    for j, col2 in enumerate(comps_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], comps_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=celltypes_df.columns,
                     columns=comps_df.columns).T

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

matrixplot(corrs, figsize=(9, 5), flip=True, scaling=False, square=True,
            colorbar_shrink=0.325, colorbar_aspect=10, title='Cell type contributions\nto tissue compartments',
            dendrogram_ratio=0.05, cbar_label="Pearsons's r", xlabel='Cell types',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Tissue compartment', rasterized=True, seed=42, linewidths=0.0, xrot=45,
            reorder_obs=True)
plt.savefig(f'data/figures/cell_types_vs_comps_exp2.svg')
plt.show()

# reference sample
adata = adata[adata.obs['sample'] == 'L2210926']

comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)
celltypes_df = adata.obsm['cell2loc']

corr_matrix = np.empty((len(celltypes_df.columns), len(comps_df.columns)))
for i, col1 in enumerate(celltypes_df.columns):
    for j, col2 in enumerate(comps_df.columns):
        corr, _ = pearsonr(celltypes_df[col1], comps_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=celltypes_df.columns,
                     columns=comps_df.columns).T

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

matrixplot(corrs, figsize=(9, 5), flip=True, scaling=False, square=True,
            colorbar_shrink=0.325, colorbar_aspect=10, title='Cell type contributions\nto tissue compartments',
            dendrogram_ratio=0.05, cbar_label="Pearsons's r", xlabel='Cell types',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Tissue compartment', rasterized=True, seed=42, linewidths=0.0, xrot=45,
            reorder_obs=True)
plt.savefig(f'data/figures/cell_types_vs_comps_ref_sample.svg')
plt.show()

