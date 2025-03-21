import numpy as np
import scanpy as sc
import pandas as pd
import decoupler as dc
import adjustText as at
from scipy.stats import zscore
import matplotlib.pyplot as plt
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet, DefaultInference
from data_analysis.utils import ensembl_id_to_gene_symbol, matrixplot


def plot_volcano_df(data, x, y, top=5, sign_thr=0.05, lFCs_thr=0.5, sign_limit=None, lFCs_limit=None,
                    color_pos='#D62728',color_neg='#1F77B4', color_null='gray', figsize=(7, 5),
                    dpi=100, ax=None, return_fig=False, s=1):

    def filter_limits(df, sign_limit=None, lFCs_limit=None):
        # Define limits if not defined
        if sign_limit is None:
            sign_limit = np.inf
        if lFCs_limit is None:
            lFCs_limit = np.inf
        # Filter by absolute value limits
        msk_sign = df['pvals'] < np.abs(sign_limit)
        msk_lFCs = np.abs(df['logFCs']) < np.abs(lFCs_limit)
        df = df.loc[msk_sign & msk_lFCs]
        return df

    # Transform sign_thr
    sign_thr = -np.log10(sign_thr)

    # Extract df
    df = data.copy()
    df['logFCs'] = df[x]
    df['pvals'] = -np.log10(df[y])

    # Filter by limits
    df = filter_limits(df, sign_limit=sign_limit, lFCs_limit=lFCs_limit)
    # Define color by up or down regulation and significance
    df['weight'] = color_null
    up_msk = (df['logFCs'] >= lFCs_thr) & (df['pvals'] >= sign_thr)
    dw_msk = (df['logFCs'] <= -lFCs_thr) & (df['pvals'] >= sign_thr)
    df.loc[up_msk, 'weight'] = color_pos
    df.loc[dw_msk, 'weight'] = color_neg
    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    df.plot.scatter(x='logFCs', y='pvals', c='weight', sharex=False, ax=ax, s=s)
    ax.set_axisbelow(True)
    # Draw sign lines
    # ax.axhline(y=sign_thr, linestyle='--', color="grey")
    # ax.axvline(x=lFCs_thr, linestyle='--', color="grey")
    # ax.axvline(x=-lFCs_thr, linestyle='--', color="grey")
    ax.axvline(x=0, linestyle='--', color="grey")
    # Plot top sign features
    # signs = df[up_msk | dw_msk].sort_values('pvals', ascending=False)
    # signs = signs.iloc[:top]

    up_signs = df[up_msk].sort_values('pvals', ascending=False)
    dw_signs = df[dw_msk].sort_values('pvals', ascending=False)
    up_signs = up_signs.iloc[:top]
    dw_signs = dw_signs.iloc[:top]
    signs = pd.concat([up_signs, dw_signs], axis=0)
    # Add labels
    ax.set_ylabel('-log10(pvals)')
    texts = []
    for x, y, s in zip(signs['logFCs'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))
    if len(texts) > 0:
        at.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ax=ax)
    if return_fig:
        return fig

def process_string(input_string):
    processed_string = input_string.replace('_', ' ')
    words = processed_string.split()
    if len(words) > 1:
        words = words[1:]
    result = ' '.join(words)
    return result

#%%

sample_folder = 'data/spatial_transcriptomics/h5ads'

# exp1
adata = sc.read_h5ad(sample_folder + '/exp1.h5ad')

adata = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()
adata.X = adata.raw.X
adata.raw = None

pdata = dc.get_pseudobulk(adata, sample_col='sample', groups_col='annotation', mode='sum', min_cells=0, min_counts=0)

inflammation = pdata[~pdata.obs['condition'].isin(['mock'])].copy()
inflammation = inflammation[~inflammation.obs['annotation'].isin(['lung_parenchyma', 'unlabeled'])].copy()

dc.plot_filter_by_expr(inflammation, group='condition', min_count=10, min_total_count=1000)
plt.show()

genes = dc.filter_by_expr(inflammation, group='condition', min_count=10, min_total_count=1000)
inflammation = inflammation[:, genes].copy()

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(adata=inflammation, design_factors=["condition"], refit_cooks=True, inference=inference,)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["condition", "OTS206", "wuhan"], inference=inference, independent_filter=True)
stat_res.summary()

results_df = stat_res.results_df
results_df.to_csv('data/spatial_transcriptomics/ots206_vs_wuhan_dgea.csv')

scale = 1.2  # 0.85
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(3*scale, 3*scale))
plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=10, ax=ax,
                   color_pos='#8b33ff', color_neg='#ff3363', s=4, sign_limit=None)
scatter = ax.collections[0]
scatter.set_rasterized(True)
# ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('-log10(p-value)')
ax.set_xlabel('logFC')
ax.set_title(f'Pseudo-bulk DEA\n<--Wuhan | OTS206-->')
plt.tight_layout()
plt.savefig('data/figures/ots206_vs_wuhan_volcano.svg', dpi=300)
plt.show()

#%% exp2

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(sample_folder + '/exp2.h5ad')

adata = ensembl_id_to_gene_symbol(adata)
adata.var_names_make_unique()
adata.X = adata.raw.X
adata.raw = None

pdata = dc.get_pseudobulk(adata, sample_col='sample', groups_col='annotation', mode='sum', min_cells=0, min_counts=0)

inflammation = pdata[~pdata.obs['condition'].isin(['mock'])].copy()
inflammation = inflammation[~inflammation.obs['annotation'].isin(['lung_parenchyma', 'unlabeled'])].copy()

dc.plot_filter_by_expr(inflammation, group='condition', min_count=100, min_total_count=4500)
plt.show()

genes = dc.filter_by_expr(inflammation, group='condition', min_count=100, min_total_count=4500)
inflammation = inflammation[:, genes].copy()

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(adata=inflammation, design_factors=["condition"], refit_cooks=True, inference=inference,)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["condition", "OTS206", "mRNA"], inference=inference, independent_filter=True)
stat_res.summary()

results_df = stat_res.results_df
results_df.to_csv('data/spatial_transcriptomics/ots206_vs_mrna_dgea.csv')

scale = 1.2  # 0.85
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(3*scale, 3*scale))
plot_volcano_df(results_df, x='log2FoldChange', y='padj', top=10, ax=ax,
                   color_pos='#8b33ff', color_neg='#ff3363', s=4, sign_limit=None)
scatter = ax.collections[0]
scatter.set_rasterized(True)
# ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('-log10(p-value)')
ax.set_xlabel('logFC')
ax.set_title(f'Pseudo-bulk DEA\n<--mRNA | OTS206-->')
plt.tight_layout()
plt.savefig('data/figures/ots206_vs_mrna_volcano.svg', dpi=300)
plt.show()


#%%
# analysis
for exp in ['ots206_vs_wuhan_dgea', 'ots206_vs_mrna_dgea']:
    results_df = pd.read_csv(f'data/spatial_transcriptomics/{exp}.csv', index_col=0)
    results_df = results_df.dropna()
    mat = results_df[['stat']].T.rename(index={'stat': 'OTS206'})

    # transcription factor
    tri = pd.read_csv('data/decoupler/tri_mouse_tfs.csv', index_col=0)
    tf_acts, tf_pvals = dc.run_ulm(mat=mat, net=tri)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = dc.plot_barplot(tf_acts, 'OTS206', top=15, vertical=True, figsize=(3, 3), return_fig=True, cmap='Spectral')
    # fig.axes[0].spines['top'].set_visible(False)
    # fig.axes[0].spines['right'].set_visible(False)
    fig.axes[0].grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
    fig.axes[0].axvline(0, color='black')
    fig.axes[0].set_axisbelow(True)
    fig.axes[0].set_title('Transcription factor activity')
    cbar = fig.axes[-1]
    cbar.set_frame_on(False)
    plt.tight_layout()
    fig.savefig(f'data/figures/{exp}_tf.svg')
    plt.show()

    # progeny
    progeny = pd.read_csv('data/decoupler/progeny_mouse_500.csv', index_col=0)
    pathway_acts, pathway_pvals = dc.run_mlm(mat=mat, net=progeny)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(1, 1)
    fig = dc.plot_barplot(pathway_acts, 'OTS206', top=25, vertical=False, return_fig=True, figsize=(3.5, 3),
                          cmap='Spectral')
    # fig.axes[0].spines['top'].set_visible(False)
    # fig.axes[0].spines['right'].set_visible(False)
    fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    fig.axes[0].axhline(0, color='black')
    fig.axes[0].set_axisbelow(True)
    fig.axes[0].set_title('Pathway activity')
    cbar = fig.axes[-1]
    cbar.set_frame_on(False)
    plt.tight_layout()
    fig.savefig(f'data/figures/{exp}_progeny.svg')
    plt.show()

#%%
# hallmarks

gene_sets_df = pd.read_csv(f'data/decoupler/msigdb.csv', index_col=0)
print(gene_sets_df['collection'].unique())
gene_sets_df = gene_sets_df[gene_sets_df['collection']=='vaccine_response']
gene_sets_df = gene_sets_df[~gene_sets_df.duplicated(['geneset', 'genesymbol'])]

# hallmarks
gene_sets_df = pd.read_csv(f'data/decoupler/hallmark_mouse.csv', index_col=0)
gene_sets_df['geneset'] = gene_sets_df['geneset'].apply(process_string)
# upregulated
top_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 0.5)]
enr_pvals = dc.get_ora_df(df=top_genes, net=gene_sets_df, source='geneset', target='genesymbol')
enr_pvals = enr_pvals.sort_values(by='Combined score', ascending=False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig = dc.plot_dotplot(enr_pvals, x='Combined score', y='Term', s='Odds ratio', c='FDR p-value',
                      scale=0.4, figsize=(6, 9), return_fig=True, cmap='rocket')
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
# fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Upregulated terms')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
plt.tight_layout()
plt.show()

# downregulated
top_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] < 0.5)]
enr_pvals = dc.get_ora_df(df=top_genes, net=gene_sets_df, source='geneset', target='genesymbol')
enr_pvals = enr_pvals.sort_values(by='Combined score', ascending=False)
plt.rcParams['svg.fonttype'] = 'none'
fig = dc.plot_dotplot(enr_pvals, x='Combined score', y='Term', s='Odds ratio', c='FDR p-value',
                      scale=0.5, figsize=(6, 9), return_fig=True, cmap='rocket')
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
# fig.axes[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
fig.axes[0].set_axisbelow(True)
fig.axes[0].set_title('Downregulated terms')
cbar = fig.axes[-1]
cbar.set_frame_on(False)
plt.tight_layout()
plt.show()
