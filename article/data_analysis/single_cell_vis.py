import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['svg.fonttype'] = 'none'

path = ("/mnt/f/10X_single_cell/annotation_final_scimilarity_celltypist/"
        "adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2024_12_17.h5ad")

adata = sc.read_h5ad(path)

# get labels

label_map = {
    'Activated B cells': '1',
    'B cells': '2',

    'CD4 T cells': '3',
    'CD8 T cells' : '4',


    'Alveolar macrophages': '5',
       'Classical monocytes': '6',
    'Dendritic cells': '7',
       'Plasmacytoid dendritic cells': '8',
    'Inflammatory Macrophages':'10',
       'Macrophages':'11',
    'Mast cells':'12',
    'Monocytes':'13',
    'Monocytes/Macrophages':'14',
       'NK cells':'15',



    'Capillary endothelial cells':'16',
       'Endothelial cells':'17',
    'EC arterial':'18',
       'Inflammed Endothelial cells':'19',
    'Lymphatic EC differentiating':'20',

    'Pericytes':'21',

    'Fibroblasts':'22',
    'Neutrophils':'23',
    'Platelets':'24',
    'Progenitor cells':'25',

       'SMG mucous':'26',
    'Type I pneumocytes':'27',
    'Type II pneumocytes':'28',
       'Vascular smooth muscle':'29',


}

# base is 80-50
color_map = {
    '1': '#19a1e6',
    '2': '#5ebeed',
    '3': '#5e5eed',
    '4': '#5aedbc',
    '5': '#6cda88',
    '6': '#3cdd51',
    '7': '#178280',
    '8': '#172982',
    '9': '#ed5e5e',
    '10': '#ed825e',
    '11': '#edb25e',
    '12': '#ed5e9c',
    '13': '#f28cb8',
    '14': '#e699c4',
    '15': '#d999e6',
    '16': '#7b53f3',
    '17': '#af75f5',
    '18': '#a947eb',
    '19': '#c788f2',
    '20': '#976ef7',
    '21': '#cbb6fb',
    '22': '#7fdd3c',
    '23': '#c32222',

    '24': '#19a1e6',
    '25': '#5ebeed',

    '26': '#5e5eed',
    '27': '#5aedbc',
    '28': '#6cda88',
    '29': '#3cdd51',
}
cell_type_color_map = {cell_type: color_map[label] for cell_type, label in label_map.items()}
# adata.obs['hires_map'] = [label_map[x] for x in adata.obs['combined_celltype_hint_celltypist']]

adata.obs['hires_map'] = [x for x in adata.obs['combined_celltype_hint_celltypist']]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
sc.pl.umap(adata, color=['hires_map'], legend_loc='on data', legend_fontoutline=1,
           palette=cell_type_color_map,
           frameon=False, ax=ax, legend_fontsize=10,
           s=40, alpha=1, show=False)
plt.show()


# primary - residual

cp = ['#8b33ff', '#ff3363']

for batch in adata.obs['condition'].cat.categories:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sc.set_figure_params(vector_friendly=True, dpi_save=300, fontsize=12)
    sc.pl.umap(adata, color=['condition'],
               palette=cp, frameon=False, groups=[batch],
               s=40, alpha=1, ax=ax, show=False)
    # plt.savefig(f'figs/manuscript/fig1/{batch}_cells.svg')
    plt.tight_layout()
    plt.show()
