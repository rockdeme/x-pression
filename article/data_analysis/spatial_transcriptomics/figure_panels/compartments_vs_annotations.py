import pandas as pd
import scanpy as sc
import chrysalis as ch
import decoupler as dc
import matplotlib.pyplot as plt

#%%
# add dictionaries for plotting

annotation_dict = {
    'atelectasis': 'Atelactasis',
    'inflammation': 'Inflammation',
    'lung_parenchyma': 'Lung parenchyma',
    'prominent_inflammation': 'Prominent inflammation',
    'unlabeled': 'Unlabeled',
}

sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')
adata.obs['annotation'] = [annotation_dict[x] for x in adata.obs['annotation']]

comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)
adata.obsm['chr_aa'] = comps_df

acts = dc.get_acts(adata, obsm_key='chr_aa')

# stacked violin plot
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
sc.set_figure_params(vector_friendly=True, dpi_save=150, fontsize=12)

fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.5))
hexcodes = ch.utils.get_hexcodes(None, 8, 42, len(adata))
dp = sc.pl.dotplot(acts, acts.var_names, groupby='annotation', swap_axes=False, dendrogram=False,
                     ax=ax, show=False, return_fig=True)
dp.style(cmap='mako_r')
axs = dp.get_axes()
mainax = axs['mainplot_ax']
mainax.set_ylabel('Annotations')
mainax.set_title('Annotations\nTissue compartment composition')

plt.setp(mainax.get_xticklabels(), rotation=0)

for leg_ax, new_text in zip(['color_legend_ax', 'size_legend_ax'],
                           ['Mean\ncompartment score', 'Fraction\nof compartment']):
    legend_ax = axs[leg_ax]
    text = legend_ax.title.get_text()
    legend_ax.title.set_text(text.replace(text, new_text))

for idx, t in enumerate(mainax.get_xticklabels()):
    t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))
plt.tight_layout()
plt.savefig(f'data/figures/compartments_vs_annots.svg')
plt.show()
