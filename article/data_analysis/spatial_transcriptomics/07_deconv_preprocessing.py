import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


#%%
# exp1
# split adatas for cell2loc
sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')

#exp1
adata_sub = adata[adata.obs['challenge'] == '-']
adata_sub = adata_sub.write_h5ad(sample_folder + '/exp1.h5ad')

#%%
# exp1 single cell
mapping_dict = {
    'AT1 cells': 'AT1 cell',
    'B ccells': 'B cell',
    'Dendritic cells': 'Dendritic cell',
    'EC aerocyte': 'EC aerocyte',
    'EC aerocyte ': 'EC aerocyte',  # Removing extra space
    'Endothelial cells': 'Endothelial cell',
    'Fibroblasts': 'Fibroblast',
    'Macrophages': 'Macrophage',
    'Mast cells': 'Mast cell',
    'Monocytes': 'Monocyte',
    'NK Cells': 'NK cell',
    'NK T cells': 'NK T cell',
    'Neutrophils': 'Neutrophil',
    'Progenitor cells': 'Progenitor cell',
    'SCOV2 AT2 cells': 'SCOV2 AT2 cell',
    'Smooth muscle cells': 'Smooth muscle cell',
    'T cells': 'T cell',
    'endotheleal cells of lymphatic vessel': 'Endothelial cell of lymphatic vessel'  # Corrected spelling
}

covid_genes = {
    'scov2_gene-GU280_gp01': 'ORF1ab',
    'scov2_gene-GU280_gp02': 'S',
    'scov2_gene-GU280_gp03': 'ORF3a',
    'scov2_gene-GU280_gp04': 'E',
    'scov2_gene-GU280_gp05': 'M',
    'scov2_gene-GU280_gp10': 'N',
    'scov2_gene-GU280_gp11': 'ORF10'
}

sc_adata_trsnfr = sc.read_h5ad('/mnt/f/10X_single_cell/Mock_OTS_WT/05_Load_and_celltypist/' +
                        'adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2025_02_21.h5ad')

sc_adata = sc.read_h5ad('/mnt/f/10X_single_cell/Mock_OTS_WT/03_Load_and_integrate/scVI/'
                        'adatas/scvi_model_replicate_no_doublets_0.1_64_2_30_64_all/' +
                        'scvi_model_replicate_no_doublets_0.1_64_2_30_64_all.h5ad')

sc_adata.obs["combined_celltype_hint_celltypist"] = sc_adata_trsnfr.obs["combined_celltype_hint_celltypist"]

sc_adata.obs['combined_celltype_hint_celltypist'] = \
    [mapping_dict[x] for x in sc_adata.obs['combined_celltype_hint_celltypist']]

# get raw counts - we don't need this in the end
raw_counts = pd.DataFrame()
for p in glob('/mnt/f/10X_single_cell/20230225_mouse_lung_pilot/*'):
    matrix_dir = f"{p}/filtered_feature_bc_matrix.h5"
    ad = sc.read_10x_h5(matrix_dir)
    ad.var_names = [x.split('__')[-1] for x in ad.var_names]
    ad.var.index = [x.split('__')[-1] for x in ad.var.index]
    ad.var_names_make_unique()
    raw_counts = pd.concat([raw_counts, ad.to_df()])

var_names = [covid_genes[x] if x in list(covid_genes.keys()) else x for x in raw_counts.columns]
raw_counts.columns = var_names
gene_set = set(sc_adata.var_names)
filtered_cols = set(var_names).intersection(gene_set)
print(f'genes in sc_adata: {len(gene_set)}\ngenes in raw: {len(var_names)}\ninteresection: {len(filtered_cols)}')
diff = gene_set.difference(var_names)
print(f'Non-matching IDs: {list(diff)[:10]}...')


sc_adata.raw = sc.AnnData(X=sc_adata.layers['raw_counts'], var=sc_adata.var, obs=sc_adata.obs)
var_df = ad.var.copy()
var_df.index = var_names

# add ensembl IDs
sc_adata.var['gene_ids'] = var_df['gene_ids']

sc_adata.write_h5ad('/mnt/f/10X_single_cell/Mock_OTS_WT/Mock_OTS_WT.h5ad')

#%%
# exp2
#split adatas for cell2loc

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(f'{sample_folder}/dataset_transferred.h5ad')

adata_sub = adata[adata.obs['challenge'] != '-']
adata_sub = adata_sub.write_h5ad(sample_folder + '/exp2.h5ad')

#%%
# exp2 single cell
mapping_dict = {
    'Activated B cells': 'Activated B cell',
    'Alveolar macrophages': 'Alveolar macrophage',
    'B cells': 'B cell',
    'CD4 T cells': 'CD4 T cell',
    'CD8 T cells': 'CD8 T cell',
    'Capillary endothelial cells': 'Capillary endothelial cell',
    'Classical monocytes': 'Classical monocyte',
    'Dendritic cells': 'Dendritic cell',
    'EC arterial': 'EC arterial',  # No change needed
    'Endothelial cells': 'Endothelial cell',
    'Fibroblasts': 'Fibroblast',
    'Inflammatory Macrophages': 'Inflammatory Macrophage',
    'Inflammed Endothelial cells': 'Inflamed Endothelial cell',  # Corrected spelling
    'Lymphatic EC differentiating': 'Lymphatic EC differentiating',  # No clear singular form
    'Macrophages': 'Macrophage',
    'Mast cells': 'Mast cell',
    'Monocytes': 'Monocyte',
    'Monocytes/Macrophages': 'Monocyte',
    'NK cells': 'NK cell',
    'Neutrophils': 'Neutrophil',
    'Pericytes': 'Pericyte',
    'Plasmacytoid dendritic cells': 'Plasmacytoid dendritic cell',
    'Platelets': 'Platelet',
    'Progenitor cells': 'Progenitor cell',
    'SMG mucous': 'SMG mucous',  # No clear change needed
    'Type I pneumocytes': 'Type I pneumocyte',
    'Type II pneumocytes': 'Type II pneumocyte',
    'Vascular smooth muscle': 'Vascular smooth muscle'  # Already singular
}

covid_genes = {
    'scov2_gene-GU280_gp01': 'ORF1ab',
    'scov2_gene-GU280_gp02': 'S',
    'scov2_gene-GU280_gp03': 'ORF3a',
    'scov2_gene-GU280_gp04': 'E',
    'scov2_gene-GU280_gp05': 'M',
    'scov2_gene-GU280_gp10': 'N',
    'scov2_gene-GU280_gp11': 'ORF10'
}

sc_adata = sc.read_h5ad('/mnt/f/10X_single_cell/E1_E6/06_Rescued_original_genes/' +
                        'adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2024_12_17_rescued.h5ad')

sc_adata.obs['combined_celltype_hint_celltypist'] = \
    [mapping_dict[x] for x in sc_adata.obs['combined_celltype_hint_celltypist']]

# get raw counts - we don't need this in the end
raw_counts = pd.DataFrame()
for p in glob('/mnt/f/10X_single_cell/20230705_mouse_lung/*'):
    matrix_dir = f"{p}/filtered_feature_bc_matrix.h5"
    ad = sc.read_10x_h5(matrix_dir)
    ad.var_names = [x.split('__')[-1] for x in ad.var_names]
    ad.var.index = [x.split('__')[-1] for x in ad.var.index]
    ad.var_names_make_unique()
    raw_counts = pd.concat([raw_counts, ad.to_df()])

var_names = [covid_genes[x] if x in list(covid_genes.keys()) else x for x in raw_counts.columns]
raw_counts.columns = var_names
gene_set = set(sc_adata.var_names)
filtered_cols = set(var_names).intersection(gene_set)
print(f'genes in sc_adata: {len(gene_set)}\ngenes in raw: {len(var_names)}\ninteresection: {len(filtered_cols)}')
diff = gene_set.difference(var_names)
print(f'Non-matching IDs: {list(diff)[:10]}...')

sc_adata.raw = sc.AnnData(X=sc_adata.layers['raw_counts'], var=sc_adata.var, obs=sc_adata.obs)
var_df = ad.var.copy()
var_df.index = var_names

# add ensembl IDs
sc_adata.var['gene_ids'] = var_df['gene_ids']

sc_adata.write_h5ad('/mnt/f/10X_single_cell/E1_E6/E1_E6.h5ad')
