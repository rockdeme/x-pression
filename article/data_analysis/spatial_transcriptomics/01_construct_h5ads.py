"""
Script for reading 10x Visium samples and converting them as h5ad files:
- metadata is added to the .obs columns
- ENSEMBL IDs used to replace gene symbols
- SCOV2 gene names are updated

"""
import os
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
from data_analysis.utils import gene_symbol_to_ensembl_id


drive = 'f'
write_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'

paths = [
    f'/mnt/{drive}/10X_visium/exp_1/rerun_combined/',
    f'/mnt/{drive}/10X_visium/exp_2/run_55_64/',
    f'/mnt/{drive}/10X_visium/covid_lung_2023-04-25/',
]

covid_genes = {
    'scov2_gene-GU280_gp01': 'ORF1ab',
    'scov2_gene-GU280_gp02': 'S',
    'scov2_gene-GU280_gp03': 'ORF3a',
    'scov2_gene-GU280_gp04': 'E',
    'scov2_gene-GU280_gp05': 'M',
    'scov2_gene-GU280_gp10': 'N',
    'scov2_gene-GU280_gp11': 'ORF10'
}

os.makedirs(write_folder, exist_ok=True)
metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

for path in paths:
    folders = glob(path + '*/')
    folders = [x for x in folders if any(sub in x for sub in list(metadata_df.index))]
    folders.sort(key=len)

    for f in tqdm(folders, desc=f'Processing folders...'):
        fname = f.split('/')[-2]
        # because of the issue with the tube labels wrong spots have been filtered out in the filtered matrix, so we
        # need to use the raw and do the filtering with the gene indices from the filtered matrix
        adata_filtered = sc.read_visium(f)
        adata_filtered = gene_symbol_to_ensembl_id(adata_filtered)

        adata = sc.read_visium(f, count_file='raw_feature_bc_matrix.h5')
        adata = gene_symbol_to_ensembl_id(adata)

        # do the filtering
        adata = adata[:, adata_filtered.var_names].copy()
        # adata = adata[adata.obs['in_tissue'] == True].copy() #  do this separately to be able to get rid of contam.

        # remove the scov2 / mm10 prefixes
        adata.var['gene_symbols'] = [covid_genes[c] if c in covid_genes.keys() else c
                                     for c in adata.var['gene_symbols']]
        # adata.var_names = [covid_genes[c] if c in covid_genes.keys() else c for c in adata.var.index]

        adata.var.index = [c.split('mm10__')[1] if 'mm10__' in c else c for c in adata.var.index]
        adata.var_names = [c.split('mm10__')[1] if 'mm10__' in c else c for c in adata.var.index]
        adata.var['gene_ids'] = [c.split('mm10__')[1] if 'mm10__' in c else c for c in adata.var['gene_ids']]
        adata.var['gene_symbols'] = [c.split('mm10__')[1] if 'mm10__' in c else c for c in adata.var['gene_symbols']]

        # add sample labels
        sample_meta = metadata_df.loc[fname]
        adata.obs['sample'] = fname
        for c in sample_meta.index:
            adata.obs[c] = sample_meta[c]

        adata.write(f'{write_folder}/{fname}_{sample_suffix}.h5ad')
