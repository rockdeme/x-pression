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
adata_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\03_Load_and_integrate\scVI\adatas\scvi_model_replicate_no_doublets_0.1_64_2_30_64_all\scvi_model_replicate_no_doublets_0.1_64_2_30_64_all.h5ad"
adata = sc.read_h5ad(adata_path)
# %%
adata

# %%
adata_annotated_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\05_Load_and_celltypist\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2025_02_21.h5ad"
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
adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\06_Rescued_original_genes\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2025_02_21_rescued.h5ad")
# %%
