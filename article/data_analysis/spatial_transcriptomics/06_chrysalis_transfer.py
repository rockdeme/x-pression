import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import numpy as np
import chrysalis as ch
import matplotlib.pyplot as plt
import archetypes as arch
from data_analysis.utils import ensembl_id_to_gene_symbol
from sklearn.decomposition import PCA
import seaborn as sns
from data_analysis.utils import plot_adatas


def transfer_archetypes(adata, model, n_pcs=20):
    adata.obsm['chr_aa'] = model.transform(adata.obsm['chr_X_pca'][:, :n_pcs])
    aa_loadings = np.dot(model.archetypes_, adata.uns['chr_pca']['loadings'][:n_pcs, :])

    if 'chr_aa' not in adata.uns.keys():
        adata.uns['chr_aa'] = {'archetypes': model.archetypes_,
                               'alphas': model.alphas_,
                               'loadings': aa_loadings,
                               'RSS': model.rss_}
    else:
        adata.uns['chr_aa']['archetypes'] = model.archetypes_
        adata.uns['chr_aa']['alphas'] = model.alphas_
        adata.uns['chr_aa']['loadings'] = aa_loadings
        adata.uns['chr_aa']['RSS'] = model.rss_
    return adata


def transfer_pcs(adata, model):
    gene_arr = np.asarray(adata[:, adata.var['spatially_variable'] == True].X.todense())
    adata.obsm['chr_X_pca'] = model.transform(gene_arr)

    if 'chr_pca' not in adata.uns.keys():
        adata.uns['chr_pca'] = {'variance_ratio': model.explained_variance_ratio_,
                                'loadings': model.components_,
                                'features': list(adata[:, adata.var['spatially_variable'] == True].var_names)}
    else:
        adata.uns['chr_pca']['variance_ratio'] = model.explained_variance_ratio_
        adata.uns['chr_pca']['loadings'] = model.components_
        adata.uns['chr_pca']['features'] = list(adata[:, adata.var['spatially_variable'] == True].var_names)
    return adata


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'filtered'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
samples.sort(key=len)

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_filtered.h5ad')
adata = ensembl_id_to_gene_symbol(adata)

ch.pca(adata)
ch.plot_explained_variance(adata)
plt.show()

ch.utils.estimate_compartments(adata, range_archetypes=(3, 20), max_iter=10)

x = adata.uns['chr_aa']['RSSs'].keys()
y = adata.uns['chr_aa']['RSSs'].values()
plt.plot(x, y)
plt.show()

ch.aa(adata, n_pcs=20, n_archetypes=8)
ch.plot(adata, sample_id='L2210926', seed=42)
plt.show()

ch.plot_weights(adata, seed=42)
plt.show()

gene_arr = np.asarray(adata[:, adata.var['spatially_variable'] == True].X.todense())
pca_model = PCA(n_components=50, svd_solver='arpack', random_state=42)
pca_model.fit(gene_arr)

model = arch.AA(n_archetypes=8, n_init=3, max_iter=200, tol=0.001, random_state=42)
model.fit(adata.obsm['chr_X_pca'][:, :20])

comps_df = pd.DataFrame(adata.obsm['chr_aa'], index=adata.obs.index)
comps_df.to_csv('data/spatial_transcriptomics/compartments_L2210926.csv')

#%%
# transfer_ad = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2212010_filtered.h5ad')
# transfer_ad = ensembl_id_to_gene_symbol(transfer_ad)
#
# transfer_ad.var['spatially_variable'] = adata.var['spatially_variable']
#
# transfer_ad = transfer_pcs(transfer_ad, pca_model)
# transfer_ad = transfer_archetypes(transfer_ad, model, n_pcs=20)
#
# ch.plot(transfer_ad, sample_id='L2212010', seed=42)
# plt.show()
#
# ch.plot_weights(adata, seed=42)
# plt.show()

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'transferred'

adatas = {}
for s in tqdm(samples):
    transfer_ad = sc.read_h5ad(s)
    sample_name = transfer_ad.obs['sample'].values[0]

    transfer_ad = ensembl_id_to_gene_symbol(transfer_ad)
    transfer_ad.var['spatially_variable'] = adata.var['spatially_variable']
    transfer_ad = transfer_pcs(transfer_ad, pca_model)
    transfer_ad = transfer_archetypes(transfer_ad, model, n_pcs=20)
    fname = transfer_ad.obs['sample'][0]
    transfer_ad.write(f'{sample_folder}/{fname}_{sample_suffix}.h5ad')

    comps_df = pd.DataFrame(transfer_ad.obsm['chr_aa'], index=transfer_ad.obs.index)
    comps_df.to_csv(f'data/spatial_transcriptomics/compartments_{sample_name}.csv')

    ch.plot(transfer_ad, sample_id=sample_name, seed=42)
    plt.title(sample_name)
    plt.show()
    adatas[sample_name] = transfer_ad

adata = ch.integrate_adatas(list(adatas.values()), list(adatas.keys()), sample_col='chr_sample_id')
adata.write(f'{sample_folder}/dataset_transferred.h5ad')

#%%
sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')

ch.plot_samples(adata, rows=4, cols=6, dim=8, sample_col='chr_sample_id', show_title=True, seed=42)
plt.show()

#%%
sample_order = list(adata.obs['chr_sample_id'].cat.categories)

# plot sample IDs and H&E for all samples
sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
samples.sort(key=len)
sample_names = [p.split('/')[-1].split('_pre')[0] for p in samples]
sample_adatas = {k: sc.read_h5ad(v) for k, v in zip(sample_names, samples) if k in sample_order}
sample_adatas = {key: sample_adatas[key] for key in sample_order if key in sample_adatas}

plot_adatas(sample_adatas.values(), color=None, alpha=0, rows=4, cols=6)
plt.show()

#%%
# sanity check
transfer_ad = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2212010_filtered.h5ad')
transfer_ad = ensembl_id_to_gene_symbol(transfer_ad)

ch.pca(transfer_ad)
ch.aa(transfer_ad, n_pcs=20, n_archetypes=8)
ch.plot(transfer_ad, sample_id='L2212010', seed=42)
plt.show()

ch.plot_weights(transfer_ad, seed=42)
plt.show()

sc.pl.spatial(transfer_ad, color=['N', 'M'], use_raw=False, cmap='viridis', s=12, alpha_img=0.8)
plt.show()
sc.pl.spatial(adata, color=['N', 'M'], use_raw=False, cmap='viridis', s=12, alpha_img=0.8)
plt.show()

ch.plot_weights(adata)
plt.show()

#%%
# sample qc

total_counts = {}
n_genes = {}
log_counts = {}

for s in tqdm(samples):
    transfer_ad = sc.read_h5ad(s)
    sample_name = transfer_ad.obs['sample'].values[0]

    counts = transfer_ad.obs['total_counts']
    total_counts[sample_name] = counts
    lcounts = transfer_ad.obs['log1p_total_counts']
    log_counts[sample_name] = lcounts
    genes = transfer_ad.obs['n_genes_by_counts']
    n_genes[sample_name] = genes

    ch.plot(transfer_ad, sample_id=sample_name, seed=42)
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    total_counts_df = pd.DataFrame(total_counts)
    sns.boxplot(total_counts_df, ax=axs[0])
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
    axs[0].set_title('Counts per spot')

    total_counts_sum_df = total_counts_df.sum(axis=0)
    sns.barplot(total_counts_sum_df, ax=axs[1])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
    axs[1].set_title('Total counts')

    n_genes_df = pd.DataFrame(n_genes)
    sns.boxplot(n_genes_df, ax=axs[2])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90)
    axs[2].set_title('N genes per spot')

    log_counts_df = pd.DataFrame(log_counts)
    sns.boxplot(log_counts_df, ax=axs[3])
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=90)
    axs[3].set_title('Log counts per spot')

    plt.tight_layout()
    plt.show()
