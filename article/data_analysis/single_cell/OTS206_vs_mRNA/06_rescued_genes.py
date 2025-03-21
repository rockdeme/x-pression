print("Loading packages...")
#!pip install gdown GPUtil psutil leidenalg scrublet scvi-tools
import scanpy as sc
import pandas as pd
import numpy as np
import scvi
import torch
import anndata
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, time, glob, csv, json, random, pickle
from tqdm import tqdm as tqdm

# %%
adata_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\03_Load_and_integrate\adatas\SCVI\adatas_replicate_0.1_128_2_50_64_all\scvi_model_replicate_all_no_doublets.h5ad"
adata = sc.read_h5ad(adata_path)

# %%
adata_annotated_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\05_Load_and_celltypist\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2024_12_17.h5ad"
adata_annotated = sc.read_h5ad(adata_annotated_path)
# %%
adata_annotated

# %%
# Add column from celltypist and scimilarity to the adata
adata.obs['combined_celltype_hint_celltypist'] = adata_annotated.obs['combined_celltype_hint_celltypist']
#adata.obs['celltypist_last_geneset_used_predicted_labels'] = adata_annotated.obs['predicted_labels']
# %%
sc.pl.umap(adata, color=['combined_celltype_hint_celltypist'], ncols=1,size=55,alpha=0.75)

# %% # Save the adata
adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\06_Rescued_original_genes\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2024_12_17_rescued.h5ad")
# %%
