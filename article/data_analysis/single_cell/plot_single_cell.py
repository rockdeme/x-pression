import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


adata = sc.read_h5ad('data/single_cell/single_cell_Mock_OTS_WT.h5ad')


# get labels

label_map = {
    'AT1 cell': '1',
    'EC aerocyte': '2',
    'Progenitor cell': '3',
    'SCOV2 AT2 cell': '4',

    'Endothelial cell': '5',
    'Endothelial cell of lymphatic vessel': '6',
    'Fibroblast': '7',
    'Smooth muscle cell': '8',

    'B cell': '9',
    'Dendritic cell': '10',
    'Macrophage': '11',
    'Mast cell': '12',
    'Monocyte': '13',
    'Neutrophil': '14',
    'NK cell': '15',
    'NK T cell': '16',
    'T cell': '17',
}

# base is 80-50
color_map = {
    '1': '#a947eb',
    '2': '#c788f2',
    '3': '#976ef7',
    '4': '#cbb6fb',
    '5': '#19a1e6',
    '6': '#5ebeed',
    '7': '#5e5eed',
    '8': '#5aedbc',
    '9': '#6cda88',
    '10': '#3cdd51',
    '11': '#178280',
    '12': '#172982',
    '13': '#ed5e5e',
    '14': '#ed825e',
    '15': '#edb25e',
    '16': '#ed5e9c',
    '17': '#f28cb8',
    '18': '#e699c4',
    '19': '#d999e6',
    '20': '#7b53f3',
    '21': '#af75f5',
    '22': '#7fdd3c',
    '23': '#c32222',
}

adata.obs['cell_type_map'] = [label_map[x] for x in adata.obs['combined_celltype_hint_celltypist']]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample", layer='log_counts')
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc.set_figure_params(vector_friendly=True, dpi_save=300)
sc.pl.umap(adata, color=['cell_type_map'], legend_loc=None, legend_fontoutline=1,
        palette=color_map,
        frameon=False, ax=ax, legend_fontsize=10,
        s=40, alpha=1, show=False)
ax.set_title(None)
plt.tight_layout()
plt.savefig('data/figures/sc_umap_exp1.svg')
plt.show()

long_names = [f'{n} | {l}' for n, l in zip(list(label_map.values()), list(label_map.keys()))]
label_dict = {k: v for k, v in zip(long_names, list(color_map.values()))}

legend_patches = [mpatches.Patch(color=label_dict[key], label=key) for key in label_dict]

fig, ax = plt.subplots(figsize=(5, 6))
sc.set_figure_params(vector_friendly=True, dpi_save=300)
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10,
                              markerfacecolor=label_dict[key]) for key in label_dict],
          loc='center', title=None, frameon=False)
ax.axis('off')
plt.savefig('data/figures/sc_umap_exp1_legends.svg')
plt.show()

# 3 conditions

cp = ['#8b33ff', '#ff3363', '#ff6933']
title_map = {'pilot_OTS206': 'OTS206', 'pilot_mock': 'Mock', 'pilot_wt': 'Wuhan'}
fig, axs = plt.subplots(1, 3, figsize=(11*0.8, 4*0.8))
for idx, batch in enumerate(adata.obs['sample'].cat.categories):
    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    sc.pl.umap(adata, color=['sample'], title=title_map[batch],
               palette=cp, frameon=False, groups=[batch],
               s=40, alpha=1, ax=axs[idx], show=False, legend_loc=None)
plt.tight_layout()
plt.savefig('data/figures/sc_umap_exp1_conditions.svg')
plt.show()

#%%

adata = sc.read_h5ad('data/single_cell/single_cell_E1_E6.h5ad')

# get labels

label_map = {

    'Activated B cell': '1',
    'Alveolar macrophage': '2',
    'B cell': '3',
    'CD4 T cell': '4',
    'CD8 T cell': '5',
    'Classical monocyte': '6',
    'Dendritic cell': '7',
    'Inflammatory Macrophage': '8',
    'Macrophage': '9',
    'Mast cell': '10',
    'Monocyte': '11',
    'NK cell': '12',
    'Neutrophil': '13',
    'Plasmacytoid dendritic cell': '14',

    'Capillary endothelial cell': '15',
    'EC arterial': '16',
    'Endothelial cell': '17',
    'Fibroblast': '18',
    'Inflamed Endothelial cell': '19',
    'Lymphatic EC differentiating': '20',
    'Pericyte': '21',
    'Platelet': '22',
    'Vascular smooth muscle': '23',

    'Progenitor cell': '24',
    'SMG mucous': '25',
    'Type I pneumocyte': '26',
    'Type II pneumocyte': '27',
}

# base is 80-50
color_map = {
    '1': '#a947eb',
    '2': '#c788f2',
    '3': '#976ef7',
    '4': '#cbb6fb',
    '5': '#19a1e6',
    '6': '#5ebeed',
    '7': '#5e5eed',
    '8': '#5aedbc',
    '9': '#6cda88',
    '10': '#3cdd51',
    '11': '#178280',
    '12': '#172982',
    '13': '#ed5e5e',
    '14': '#ed825e',
    '15': '#edb25e',
    '16': '#ed5e9c',
    '17': '#f28cb8',
    '18': '#e699c4',
    '19': '#d999e6',
    '20': '#7b53f3',
    '21': '#af75f5',
    '22': '#7fdd3c',
    '23': '#c32222',
    '24': '#dbce0f',
    '25': '#f3e958',
    '26': '#d2f358',
    '27': '#afdb0f',
}

adata.obs['cell_type_map'] = [label_map[x] for x in adata.obs['combined_celltype_hint_celltypist']]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

sc.pp.normalize_total(adata, layer='raw_counts')
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc.set_figure_params(vector_friendly=True, dpi_save=300)
sc.pl.umap(adata, color=['cell_type_map'], legend_loc=None, legend_fontoutline=1,
        palette=color_map,
        frameon=False, ax=ax, legend_fontsize=10,
        s=40, alpha=1, show=False)
ax.set_title(None)
plt.tight_layout()
plt.savefig('data/figures/sc_umap_exp2.svg')
plt.show()

long_names = [f'{n} | {l}' for n, l in zip(list(label_map.values()), list(label_map.keys()))]
label_dict = {k: v for k, v in zip(long_names, list(color_map.values()))}

legend_patches = [mpatches.Patch(color=label_dict[key], label=key) for key in label_dict]

fig, ax = plt.subplots(figsize=(5, 10))
sc.set_figure_params(vector_friendly=True, dpi_save=300)
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10,
                              markerfacecolor=label_dict[key]) for key in label_dict],
          loc='center', title=None, frameon=False)
ax.axis('off')
plt.savefig('data/figures/sc_umap_exp2_legends.svg')
plt.show()

# 3 conditions

cp = ['#8b33ff', '#ff3363']
adata.obs['label'] = [x + '-' + y for x, y in zip(adata.obs['condition'], adata.obs['timepoint'])]
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for idx, batch in enumerate(np.unique(adata.obs['label'])):
    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    sc.pl.umap(adata, color=['label'], title=batch,
               palette=cp, frameon=False, groups=[batch],
               s=40, alpha=1, ax=axs[idx], show=False, legend_loc=None)
plt.tight_layout()
plt.savefig('data/figures/sc_umap_exp2_conditions.svg')
plt.show()
