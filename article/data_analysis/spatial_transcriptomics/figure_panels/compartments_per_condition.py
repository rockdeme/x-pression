import numpy as np
import pandas as pd
import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis.chrysalis_dev_functions import plot_samples


# read adata
sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')

# create df from chrysalis array
comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)
# adata.obsm['chr_aa'] = comps_df
# add compartment colors
hexcodes = ch.utils.get_hexcodes(None, 8, 42, len(adata))

adata.obs['label_exp1'] = adata.obs['condition'].astype(str) + '-' + adata.obs['timepoint'].astype(str)
adata.obs['label_exp2'] = (adata.obs['condition'].astype(str) +
                          '-' + adata.obs['timepoint'].astype(str) +
                          '-' + adata.obs['challenge'].astype(str))

def get_proportion_df(adata, sample_col, condition_col):
    labels = []
    spot_nr = []
    prop_matrix = np.zeros((len(np.unique(adata.obs[sample_col])), 8))
    for idx, i in enumerate(np.unique(adata.obs[sample_col])):
        ad = adata[adata.obs[sample_col] == i]
        spot_nr.append(len(ad))
        # compartments = ad.obsm['chr_aa']
        compartments = pd.DataFrame(ad.obsm['chr_aa'],
                                    columns=[x for x in range(ad.obsm['chr_aa'].shape[1])],
                                    index=ad.obs_names)
        compartments_mean = compartments.sum(axis=0)
        compartments_prop = compartments_mean / np.sum(compartments_mean)
        prop_matrix[idx] = compartments_prop.values
        label = adata.obs[adata.obs[sample_col] == i][condition_col][0]
        labels.append(label)
    props_df = pd.DataFrame(data=prop_matrix, index=labels)
    # spot_nr = pd.Series(data=spot_nr, index=labels, name='spot_nr')
    return props_df

# subset for experiments
# experiment 1
sub_adata = adata[adata.obs['challenge'] == '-']
# define the custom order
custom_order = ['mock-2dpi', 'mock-5dpi', 'wuhan-2dpi', 'wuhan-5dpi', 'OTS206-2dpi', 'OTS206-5dpi']
# compartment proportions per group
props_df = get_proportion_df(sub_adata, 'sample', 'label_exp1')
props_df.index = props_df.index.astype('category')
props_df.index = props_df.index.reorder_categories(custom_order, ordered=True)

# barplot
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
rows = 2
cols = 4
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(2.25 * cols, 2.5 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(range(sub_adata.obsm['chr_aa'].shape[1])):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sns.barplot(sub_df, y=c, x=sub_df.index, ax=axs[idx], color=hexcodes[c])
    sns.stripplot(sub_df, y=c, x=sub_df.index, ax=axs[idx], color='#4C4C4C')
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=10)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
fig.supylabel(None)
plt.tight_layout()
plt.savefig('data/figures/exp1_chr_barplots.svg')
plt.show()

# subset anndata
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
selected_samples = ['L221201', 'L221202', 'L2212010', 'L221203', 'L221205', 'L221208']
sub_adata = sub_adata[sub_adata.obs['sample'].isin(selected_samples)].copy()
# reorder sample_id for plotting
sub_adata.obs['sample'] = sub_adata.obs['sample'].cat.reorder_categories(selected_samples, ordered=True)
for i in range(8):
    plot_samples(sub_adata, 1, 6, 8, sample_col='sample', seed=42, spot_size=2.15, rasterized=True,
                 hspace=0.5, wspace=-0.7, selected_comp=i, title_col='label_exp1', plot_size=3.0)
    plt.savefig(f'data/figures/exp1_chr_comp_{i}.svg')
    plt.show()

plot_samples(sub_adata, 1, 6, 8, sample_col='sample', seed=42, spot_size=2.15, rasterized=True,
             hspace=0.5, wspace=-0.7, title_col='label_exp1', plot_size=3.0)
plt.savefig(f'data/figures/exp1_chr_comps.svg')
plt.show()

# experiment 2
sub_adata = adata[adata.obs['challenge'] != '-']
# define the custom order
custom_order = ['mRNA-2dpi-delta', 'mRNA-5dpi-delta', 'mRNA-5dpi-mock',
                'OTS206-2dpi-delta', 'OTS206-5dpi-delta', 'OTS206-5dpi-mock']
# compartment proportions per group
props_df = get_proportion_df(sub_adata, 'sample', 'label_exp2')
props_df.index = props_df.index.astype('category')
props_df.index = props_df.index.reorder_categories(custom_order, ordered=True)

# barplot
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
rows = 2
cols = 4
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(2.25 * cols, 2.9 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(range(sub_adata.obsm['chr_aa'].shape[1])):
    axs[idx].axis('on')
    sub_df = props_df[[c]].copy()
    sns.barplot(sub_df, y=c, x=sub_df.index, ax=axs[idx], color=hexcodes[c])
    sns.stripplot(sub_df, y=c, x=sub_df.index, ax=axs[idx], color='#4C4C4C')
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Proportion')
    axs[idx].set_title(f'Compartment {c}', fontsize=10)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
fig.supylabel(None)
plt.tight_layout()
plt.savefig('data/figures/exp2_chr_barplots.svg')
plt.show()

# subset anndata
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
selected_samples = ['L2210915',  'L2210925',  'L2210927',  'L221097',  'L2210911',  'L221093']
sub_adata = sub_adata[sub_adata.obs['sample'].isin(selected_samples)].copy()
# reorder sample_id for plotting
sub_adata.obs['sample'] = sub_adata.obs['sample'].cat.reorder_categories(selected_samples, ordered=True)
for i in range(8):
    plot_samples(sub_adata, 1, 6, 8, sample_col='sample', seed=42, spot_size=2.15, rasterized=True,
                 hspace=0.5, wspace=-0.7, selected_comp=i, title_col='label_exp2', plot_size=3.0)
    plt.savefig(f'data/figures/exp2_chr_comp_{i}.svg')
    plt.show()

plot_samples(sub_adata, 1, 6, 8, sample_col='sample', seed=42, spot_size=2.15, rasterized=True,
             hspace=0.5, wspace=-0.7, title_col='label_exp2', plot_size=3.0)
plt.savefig(f'data/figures/exp2_chr_comps.svg')
plt.show()
