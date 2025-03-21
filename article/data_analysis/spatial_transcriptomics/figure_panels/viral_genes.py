import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from data_analysis.utils import spatial_plot

# read adata
covid_genes = {
    'scov2_gene-GU280_gp01': 'ORF1ab',
    'scov2_gene-GU280_gp02': 'S',
    'scov2_gene-GU280_gp03': 'ORF3a',
    'scov2_gene-GU280_gp04': 'E',
    'scov2_gene-GU280_gp05': 'M',
    'scov2_gene-GU280_gp10': 'N',
}

custom_order = ['mock-2dpi', 'mock-5dpi', 'wuhan-2dpi', 'wuhan-5dpi', 'OTS206-2dpi', 'OTS206-5dpi']

sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')
adata_viral = adata[:, adata.var_names.isin(covid_genes.keys())]
df_raw = pd.DataFrame(adata_viral.raw.X.toarray(), index=adata_viral.obs_names, columns=adata_viral.raw.var_names)
df_raw = df_raw[list(covid_genes.keys())]
adata_viral.X = df_raw.sum(axis=1)
sc.pp.normalize_total(adata_viral, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
sc.pp.log1p(adata_viral)
df_raw = pd.DataFrame(adata_viral.X.toarray(), index=adata_viral.obs_names, columns=adata_viral.var_names)
adata.raw = None
df = adata.to_df()[list(covid_genes.keys())]
adata.obs['viral_expression'] = df_raw

# experiment 1
# viral gene expression across conditions
adata.obs['condition_exp1'] = adata.obs['condition'].astype(str) + '-' + adata.obs['timepoint'].astype(str)
df_melted = adata.obs.melt(id_vars=['condition_exp1', 'challenge'], value_vars='viral_expression')
df_melted = df_melted[df_melted['challenge'] == '-']
df_melted['condition_exp1'] = pd.Categorical(df_melted['condition_exp1'], categories=custom_order, ordered=True)

plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(1, 1, figsize=(2.5, 3))
axs.axis('off')
axs.axis('on')
sns.boxplot(df_melted, y=df_melted['value'], x=df_melted['condition_exp1'], ax=axs, color='#1ae6b3')
axs.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
axs.set_axisbelow(True)
axs.set_ylabel('Expression')
axs.set_title(f'Total viral expression', fontsize=10)
axs.set_xlabel(None)
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
fig.supylabel(None)
plt.tight_layout()
plt.show()

# spatial plot
# subset anndata
selected_samples = ['L221201', 'L221202', 'L2212010', 'L221203', 'L221205', 'L221208']
sub_adata = adata[adata.obs['sample'].isin(selected_samples)].copy()
# reorder sample_id for plotting
sub_adata.obs['sample'] = sub_adata.obs['sample'].cat.reorder_categories(selected_samples, ordered=True)
spatial_plot(sub_adata, 1, 6, 'viral_expression', cmap='mako', sample_col='sample', s=12,
             title=None, suptitle='asd', alpha_img=0.8, colorbar_label='Cell density',
             colorbar_aspect=5, colorbar_shrink=0.25, wspace=-0.1, subplot_size=3, alpha_blend=True,
             x0=0.3, suptitle_fontsize=15, topspace=0.8, bottomspace=0.05, leftspace=0.1, rightspace=0.9)
plt.show()


viral_genes = ['ORF1ab', 'S', 'ORF3a', 'E', 'M', 'N']

# subset for experiments
# experiment 1
sub_adata = adata[adata.obs['challenge'] == '-']
# define the custom order

df = sub_adata.to_df()[list(covid_genes.keys())]
sub_adata.obs['viral_expression'] = df.sum(axis=1)
df['condition'] = sub_adata.obs['condition'].astype(str) + '-' + sub_adata.obs['timepoint'].astype(str)
df_melted = df.melt(id_vars='condition', var_name='gene', value_name='expression')
df_melted['condition'] = pd.Categorical(df_melted['condition'], categories=custom_order, ordered=True)

rows = 2
cols = 3
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(rows, cols, figsize=(2.25 * cols, 3 * rows))
axs = axs.flatten()
for a in axs:
    a.axis('off')
for idx, c in enumerate(covid_genes.items()):
    axs[idx].axis('on')
    sub_df = df_melted[df_melted['gene'] == c[0]]
    sns.boxplot(sub_df, y=sub_df['expression'], x=sub_df['condition'], ax=axs[idx], color='#1ae6b3')
    # sns.stripplot(sub_df, y=sub_df['expression'], x=sub_df['condition'], ax=axs[idx], color='#4C4C4C')
    # axs.set_ylim(0, 0.5)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Expression')
    axs[idx].set_title(f'{c[1]}', fontsize=10)
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=90)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
plt.suptitle('Viral gene expression after infection')
fig.supylabel(None)
plt.tight_layout()
plt.show()

from data_analysis.utils import spatial_plot

spatial_plot(adata, )
