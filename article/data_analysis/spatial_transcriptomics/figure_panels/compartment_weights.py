import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt
from data_analysis.chrysalis_dev_functions import plot_weights, plot_matrix


# read adata
sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')

expression_df = ch.get_compartment_df(adata)

top_indices = expression_df.apply(lambda col: col.nlargest(15).index)
for c in top_indices.columns:
    print(top_indices[c].values)


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plot_weights(adata, ncols=4, seed=42, w=0.6, h=0.7, top_genes=15, title_size=10)
plt.savefig('data/figures/chr_weight_barplots.svg')
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plot_matrix(adata, figsize=(3.5, 7.5), flip=False, scaling=True, square=False,
            colorbar_shrink=0.2, colorbar_aspect=10, num_genes=5,
            dendrogram_ratio=0.05, cbar_label='Z-scored gene contribution', xlabel='Tissue compartment',
            ylabel='Top contributing genes per compartment', rasterized=True, seed=42)
plt.savefig('data/figures/chr_weight_heatmap.svg')
plt.show()

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')
ch.plot(adata, sample_id='L2210926', seed=42, figsize=(10, 10), spot_size=1.15, rasterized=True)
plt.savefig('data/figures/L2210926_chr_plot.svg')
plt.show()
