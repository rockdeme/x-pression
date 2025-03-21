import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from glob import glob
import chrysalis as ch
import decoupler as dc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from data_analysis.utils import ensembl_id_to_gene_symbol, matrixplot
from scipy.stats import zscore


# pathway activities

sample_folder = 'data/spatial_transcriptomics/h5ads'

# exp1
adata = sc.read_h5ad(sample_folder + '/exp1.h5ad')

adta = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

progeny = pd.read_csv('data/spatial_transcriptomics/progeny_mouse_500.csv', index_col=0)
dc.run_mlm(mat=adata, net=progeny, source='source', target='target', weight='weight', verbose=True, use_raw=False)

acts = dc.get_acts(adata, obsm_key='mlm_estimate')
acts_df = acts.to_df()

corr_m = pd.concat([compartment_df, acts_df], axis=1).corr()
corr_m = corr_m.drop(index=acts_df.columns, columns=compartment_df.columns)
corr_m.index = [str(x) for x in range(8)]
hexcodes = ch.utils.get_hexcodes(None, 8, 42, 1)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(corr_m, figsize=(5.0, 5), flip=True, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.1, cbar_label="Score", xlabel='Pathways',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=42, reorder_obs=True,
            color_comps=True, xrot=90, ha='center')
plt.savefig(f'data/figures/pathways_correlation_exp1.svg')
plt.show()

# pathway activity mean per condition
adata.obs['label_exp1'] = adata.obs['condition'].astype(str) + '-' + adata.obs['timepoint'].astype(str)
custom_order = ['mock-2dpi', 'mock-5dpi', 'wuhan-2dpi', 'wuhan-5dpi', 'OTS206-2dpi', 'OTS206-5dpi']

pathways_matrix = np.zeros((len(np.unique(adata.obs['label_exp1'])), len(np.unique(progeny['source']))))
for idx, s in enumerate(custom_order):
    ad = adata[adata.obs['label_exp1'] == s].copy()
    pathways = ad.obsm['mlm_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=custom_order,
                           columns=pathways.columns)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(pathways_df.T, figsize=(5.0, 5), flip=False, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.0, cbar_label="Score", xlabel='Pathways',
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=42, reorder_obs=False,
            color_comps=False, xrot=90, ha='center')
plt.savefig(f'data/figures/pathways_condition_exp1.svg')
plt.show()

acts_obs = acts.obs.copy()
acts_obs['JAK-STAT'] = acts_df['JAK-STAT']
acts_obs['label_exp1'] = adata.obs['label_exp1']

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axs = plt.subplots(1, 1, figsize=(2.0, 3.0))
axs.axis('on')
axs.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
axs.set_axisbelow(True)
sns.boxplot(data=acts_obs, x='label_exp1', y='JAK-STAT',
               # scale='width',
               palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
               order=custom_order, ax=axs, showfliers=False)
# sns.stripplot(data=pw_df, x='condition_cat', y=pw, jitter=True,
#                order=order, color='black', size=2, alpha=.1)
axs.set_ylabel(None)
axs.set_title(f'JAK-STAT activity')
axs.set_xlabel(None)
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'data/figures/jak_stat_exp1.svg')
plt.show()

#%% exp2

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(sample_folder + '/exp2.h5ad')

adta = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

progeny = pd.read_csv('data/spatial_transcriptomics/progeny_mouse_500.csv', index_col=0)
dc.run_mlm(mat=adata, net=progeny, source='source', target='target', weight='weight', verbose=True, use_raw=False)

acts = dc.get_acts(adata, obsm_key='mlm_estimate')
acts_df = acts.to_df()

corr_m = pd.concat([compartment_df, acts_df], axis=1).corr()
corr_m = corr_m.drop(index=acts_df.columns, columns=compartment_df.columns)
corr_m.index = [str(x) for x in range(8)]
hexcodes = ch.utils.get_hexcodes(None, 8, 42, 1)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(corr_m, figsize=(5.0, 5), flip=True, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.1, cbar_label="Score", xlabel='Pathways',
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=42, reorder_obs=True,
            color_comps=True, xrot=90, ha='center')
plt.savefig(f'data/figures/pathways_correlation_exp2.svg')
plt.show()

# pathway activity mean per condition
adata.obs['label_exp2'] = (adata.obs['condition'].astype(str) +
                          '-' + adata.obs['timepoint'].astype(str) +
                          '-' + adata.obs['challenge'].astype(str))
custom_order = ['mRNA-2dpi-delta', 'mRNA-5dpi-delta', 'mRNA-5dpi-mock',
                'OTS206-2dpi-delta', 'OTS206-5dpi-delta', 'OTS206-5dpi-mock']

pathways_matrix = np.zeros((len(np.unique(adata.obs['label_exp2'])), len(np.unique(progeny['source']))))
for idx, s in enumerate(custom_order):
    ad = adata[adata.obs['label_exp2'] == s].copy()
    pathways = ad.obsm['mlm_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)


pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=custom_order,
                           columns=pathways.columns)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(pathways_df.T, figsize=(5.35, 5), flip=False, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.0, cbar_label="Score", xlabel='Pathways',
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=42, reorder_obs=False,
            color_comps=False, xrot=90, ha='center')
plt.savefig(f'data/figures/pathways_condition_exp2.svg')
plt.show()

acts_obs = acts.obs.copy()
acts_obs['JAK-STAT'] = acts_df['JAK-STAT']
acts_obs['label_exp2'] = adata.obs['label_exp2']

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axs = plt.subplots(1, 1, figsize=(1.75, 3.5))
axs.axis('on')
axs.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
axs.set_axisbelow(True)
sns.boxplot(data=acts_obs, x='label_exp2', y='JAK-STAT',
               # scale='width',
               palette=['#92BD6D', '#E9C46B', '#F2A360', '#E66F4F', '#4BC9F1', '#4993F0', '#435FEE'],
               order=custom_order, ax=axs, showfliers=False)
# sns.stripplot(data=pw_df, x='condition_cat', y=pw, jitter=True,
#                order=order, color='black', size=2, alpha=.1)
axs.set_ylabel(None)
axs.set_title(f'JAK-STAT activity')
axs.set_xlabel(None)
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'data/figures/jak_stat_exp2.svg')
plt.show()

#%%
# hallmarks
# exp1

sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(sample_folder + '/exp1.h5ad')

adta = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

# get gene sets - hallmark_mouse
gene_sets_df = pd.read_csv('data/spatial_transcriptomics/hallmark_mouse.csv', index_col=0)
dc.run_ora(mat=adata, net=gene_sets_df, source='geneset', target='genesymbol', verbose=True, use_raw=False)
acts = dc.get_acts(adata, obsm_key='ora_estimate')
acts_df = acts.to_df()

acts_v = acts_df.values.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts_df.values[~np.isfinite(acts_df.values)] = max_e

adata.obsm['ora_estimate'] = acts_df
pvals = dc.get_acts(adata, obsm_key='ora_pvals')
pvals_df = pvals.to_df()

signif_df = 0.05 > pvals_df
signif_df = signif_df.sum(axis=0)
signif_df = pd.DataFrame(data=signif_df, columns=['num_significant'])
signif_df['score_sum'] = acts_df.sum(axis=0)

plt.hist(signif_df['num_significant'])
plt.show()

signif_sets = signif_df[signif_df['num_significant'] > 20000].index.tolist()

# add conditions
adata.obs['label_exp1'] = adata.obs['condition'].astype(str) + '-' + adata.obs['timepoint'].astype(str)
custom_order = ['mock-2dpi', 'mock-5dpi', 'wuhan-2dpi', 'wuhan-5dpi', 'OTS206-2dpi', 'OTS206-5dpi']

pathways_matrix = np.zeros((len(np.unique(adata.obs['label_exp1'])), len(adata.obsm['ora_estimate'].columns)))
for idx, s in enumerate(custom_order):
    ad = adata[adata.obs['label_exp1'] == s]
    pathways = ad.obsm['ora_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=custom_order,
                           columns=pathways.columns)
pvals = dc.get_acts(adata, obsm_key='ora_pvals')
pvals_df = pvals.to_df()

pathways_df = pathways_df[signif_sets]
pathways_df.columns = [' '.join(x.split('_')) for x in pathways_df.columns]
pathways_df.columns = [x.split('HALLMARK ')[-1] for x in pathways_df.columns]

# df = rank_sources_groups(acts, groupby='sample_id', reference='rest', method='t-test_overestim_var')

z_score_df = pathways_df.copy()
for c in z_score_df.columns:
    z_score_df[c] = zscore(z_score_df[c])

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(z_score_df.T, figsize=(8, 6), flip=False, scaling=False, square=True,
            colorbar_shrink=0.175, colorbar_aspect=10, title='Hallmarks',
            dendrogram_ratio=0.1, cbar_label="Z-scaled\nscore", xlabel=None,
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=87, reorder_obs=False,
            color_comps=False, xrot=90, ha='center')
plt.savefig(f'data/figures/hallmarks_condition_exp1.svg')
plt.show()

# correlations
enrich_df = acts_df[signif_sets]
enrich_df.columns = [' '.join(x.split('_')) for x in enrich_df.columns]
enrich_df.columns = [x.split('HALLMARK ')[-1] for x in enrich_df.columns]

corr_m = pd.concat([compartment_df, enrich_df], axis=1).corr()
corr_m = corr_m.drop(index=enrich_df.columns, columns=compartment_df.columns)
corr_m.index = [str(x) for x in range(8)]
hexcodes = ch.utils.get_hexcodes(None, 8, 42, 1)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(corr_m, figsize=(8, 6), flip=True, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.05, cbar_label="Score", xlabel=None,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=42, reorder_obs=True,
            color_comps=True, xrot=90, ha='center')
plt.savefig(f'data/figures/hallmarks_correlation_exp1.svg')
plt.show()

#%%
# exp2

sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(sample_folder + '/exp2.h5ad')

adta = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()

compartments = adata.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=adata.obs.index)

# get gene sets - hallmark_mouse
gene_sets_df = pd.read_csv('data/spatial_transcriptomics/hallmark_mouse.csv', index_col=0)
dc.run_ora(mat=adata, net=gene_sets_df, source='geneset', target='genesymbol', verbose=True, use_raw=False)
acts = dc.get_acts(adata, obsm_key='ora_estimate')
acts_df = acts.to_df()

acts_v = acts_df.values.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts_df.values[~np.isfinite(acts_df.values)] = max_e

adata.obsm['ora_estimate'] = acts_df
pvals = dc.get_acts(adata, obsm_key='ora_pvals')
pvals_df = pvals.to_df()

signif_df = 0.05 > pvals_df
signif_df = signif_df.sum(axis=0)
signif_df = pd.DataFrame(data=signif_df, columns=['num_significant'])
signif_df['score_sum'] = acts_df.sum(axis=0)

plt.hist(signif_df['num_significant'])
plt.show()

signif_sets = signif_df[signif_df['num_significant'] > 20000].index.tolist()

# add conditions
adata.obs['label_exp2'] = (adata.obs['condition'].astype(str) +
                          '-' + adata.obs['timepoint'].astype(str) +
                          '-' + adata.obs['challenge'].astype(str))
custom_order = ['mRNA-2dpi-delta', 'mRNA-5dpi-delta', 'mRNA-5dpi-mock',
                'OTS206-2dpi-delta', 'OTS206-5dpi-delta', 'OTS206-5dpi-mock']

pathways_matrix = np.zeros((len(np.unique(adata.obs['label_exp2'])), len(adata.obsm['ora_estimate'].columns)))
for idx, s in enumerate(custom_order):
    ad = adata[adata.obs['label_exp2'] == s]
    pathways = ad.obsm['ora_estimate']
    ad.obs[pathways.columns] = pathways
    pathways_matrix[idx] = pathways.mean(axis=0)

pathways_df = pd.DataFrame(data=pathways_matrix,
                           index=custom_order,
                           columns=pathways.columns)
pvals = dc.get_acts(adata, obsm_key='ora_pvals')
pvals_df = pvals.to_df()

pathways_df = pathways_df[signif_sets]
pathways_df.columns = [' '.join(x.split('_')) for x in pathways_df.columns]
pathways_df.columns = [x.split('HALLMARK ')[-1] for x in pathways_df.columns]

# df = rank_sources_groups(acts, groupby='sample_id', reference='rest', method='t-test_overestim_var')

z_score_df = pathways_df.copy()
for c in z_score_df.columns:
    z_score_df[c] = zscore(z_score_df[c])

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(z_score_df.T, figsize=(8.5, 6), flip=False, scaling=False, square=True,
            colorbar_shrink=0.175, colorbar_aspect=10, title='Hallmarks',
            dendrogram_ratio=0.1, cbar_label="Z-scaled\nscore", xlabel=None,
            cmap=sns.diverging_palette(325, 145, l=60, s=80, center="dark", as_cmap=True),
            ylabel=None, rasterized=True, seed=87, reorder_obs=False,
            color_comps=False, xrot=90, ha='center')
plt.savefig(f'data/figures/hallmarks_condition_exp2.svg')
plt.show()

# correlations
enrich_df = acts_df[signif_sets]
enrich_df.columns = [' '.join(x.split('_')) for x in enrich_df.columns]
enrich_df.columns = [x.split('HALLMARK ')[-1] for x in enrich_df.columns]

corr_m = pd.concat([compartment_df, enrich_df], axis=1).corr()
corr_m = corr_m.drop(index=enrich_df.columns, columns=compartment_df.columns)
corr_m.index = [str(x) for x in range(8)]
hexcodes = ch.utils.get_hexcodes(None, 8, 42, 1)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(corr_m, figsize=(8, 6), flip=True, scaling=False, square=True,
            colorbar_shrink=0.25, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.05, cbar_label="Score", xlabel=None,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=42, reorder_obs=True,
            color_comps=True, xrot=90, ha='center')
plt.savefig(f'data/figures/hallmarks_correlation_exp2.svg')
plt.show()