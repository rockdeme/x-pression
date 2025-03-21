import os
import  numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import  seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from data_analysis.utils import density_scatter, parse_visiopharm_xml, map_visium_to_visiopharm, plot_adatas

#%%
sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')

total_counts = {}
n_genes = {}
log_counts = {}

for s in tqdm(samples):
    adata = sc.read_h5ad(s)
    adata = adata[adata.obs['in_tissue'] == 1]
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sample_name = adata.obs['sample'][0]
    counts = adata.obs['total_counts']
    total_counts[sample_name] = counts
    lcounts = adata.obs['log1p_total_counts']
    log_counts[sample_name] = lcounts
    genes = adata.obs['n_genes_by_counts']
    n_genes[sample_name] = genes

    sc.pl.spatial(adata, img_key="hires", color=["total_counts", "log1p_total_counts", 'n_genes_by_counts'], size=1.6,
                  cmap='viridis', alpha=0.9, show=False)
    plt.suptitle(sample_name)
    plt.savefig(sample_folder + f'/{sample_name}_qc.png')
    plt.close()

total_counts_df = pd.DataFrame(total_counts)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.boxplot(total_counts_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(sample_folder + '/counts_plot.png')
plt.show()

total_counts_sum_df = total_counts_df.sum(axis=0)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.barplot(total_counts_sum_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(sample_folder + '/counts_plot_sum.png')
plt.show()

n_genes_df = pd.DataFrame(n_genes)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.boxplot(n_genes_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(sample_folder + '/genes_plot.png')
plt.show()

log_counts_df = pd.DataFrame(log_counts)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.boxplot(log_counts_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(sample_folder + '/log_counts_plot.png')
plt.show()

#%%
# manual iterative QC
for s in samples:
    s = samples[20]
    print(s)
    adata = sc.read_h5ad(s)
    adata = adata[adata.obs['in_tissue'] == 1]
    adata.raw = adata

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    density_scatter(x=adata.obs['total_counts'], y=adata.obs['n_genes_by_counts'], cmap='viridis')
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0], bins=40)  # Use sns.histplot instead of distplot
    max_bin = 10000
    bins = np.linspace(0, max_bin, 30)  # Create bins up to 2000
    sns.histplot(adata.obs["total_counts"], bins=bins, kde=False, ax=axs[1])  # Use sns.histplot instead of distplot
    axs[1].set_xlim(0, max_bin)
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=40, ax=axs[2])
    plt.show()

    sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"], size=1.6,
                  cmap='viridis', alpha=0.9)

    sc.pp.filter_cells(adata, min_counts=2500)
    sc.pp.filter_cells(adata, min_genes=0)

    sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"], size=1.6,
                  cmap='viridis', alpha=0.9)

#%%
# plot sample IDs and H&E for all samples
sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
samples.sort(key=len)
sample_names = [p.split('/')[-1].split('_')[0] for p in samples]

# get samples that are not excluded
metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)
metadata_df = metadata_df[metadata_df['exclude'] != 'yes']

included_samples = list(metadata_df.index)

sample_adatas = {k: sc.read_h5ad(v) for k, v in zip(sample_names, samples) if k in included_samples}

plot_adatas(sample_adatas.values(), color=None, alpha=0, rows=4, cols=6)
plt.show()
