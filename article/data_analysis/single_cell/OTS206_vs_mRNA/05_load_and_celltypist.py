# %%
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
#import pymde
import tempfile
import leidenalg
import GPUtil
import psutil
import glob, re, os
from matplotlib.colors import rgb2hex
#import celltypist
from scvi.model.utils import mde
import scrublet as scr
from sklearn.metrics import adjusted_rand_score
scvi.settings.seed = 2207
from scipy.io import mmread
import sys
# import scarches as sca
# from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import gdown
#from ray import tune
#from scvi import autotune
import jax
import jax.numpy as jnp
from flax.core import freeze
#import ray 
import os, sys, time, glob, csv, json, random, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import scvi
import requests
import pandas as pd
from biomart import BiomartServer
from io import StringIO
from tqdm import tqdm as tqdm


# %%
import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get the number of GPUs
device_count = pynvml.nvmlDeviceGetCount()

# Iterate over each GPU and print its status
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
    print(f"GPU {i}: {name}")
    print(f"  Memory Total: {memory_info.total / (1024 ** 2)} MB")
    print(f"  Memory Used: {memory_info.used / (1024 ** 2)} MB")
    print(f"  Memory Free: {memory_info.free / (1024 ** 2)} MB")
    print(f"  GPU Utilization: {utilization.gpu}%")
    print(f"  Memory Utilization: {utilization.memory}%")
    print()

# Shutdown NVML
pynvml.nvmlShutdown()

# %%
# Check the number of CPUS, memory and resources available to this session.
print("CPUs:", psutil.cpu_count())
print("Memory:", psutil.virtual_memory())
print("Resources:", GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[]))


# %%
# Load the adata
adata = sc.read_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\04_Load_and_scimilarity\adata_preprocessed_integrated_scVI_annotated_scimilarity.h5ad")


# %%
# Add celltypist
from celltypist import models
import celltypist
#!pip install celltypist
# %%

# they are all stored in 
models.models_path
models.download_models(force_update = True) # force update to get the latest version will overwrite the existing old models

# %%
models.models_description()
# %%
#Autopsy_COVID19_Lung.pkl	
#Cells_Lung_Airway.pkl	
#Human_Lung_Atlas.pkl	
#Lethal_COVID19_Lung.pkl	
# drop the majority_voting column from adata.obs
#adata.obs.drop(columns = "majority_voting", inplace = True)

model = models.Model.load(model = "Human_Lung_Atlas.pkl")
print(model)
model.cell_types

# %%
adata
# %%
# Some cells were duplicated for some reason in the adata.
adata_duplicated_cells = adata.obs.index.duplicated(keep = False)
adata_duplicated_cells
# %%
adata.obs['adata_duplicated_cells'] = np.where(adata_duplicated_cells, 'duplicated', 'non-duplicated')
sc.pl.umap(adata, color = 'adata_duplicated_cells')
# %%
adata

# %%
# Remove duplicated cells
non_duplicates = ~adata.obs.index.duplicated(keep='first')
adata = adata[non_duplicates].copy()  
# adata = adata[~adata.obs['adata_duplicated_cells']].copy()
# %%
np.max(adata.X)
# %%
np.max(adata.layers["raw_counts"])
# %%
adata.X = adata.layers["raw_counts"]
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)


# %%
predictions = celltypist.annotate(adata, model = 'Human_Lung_Atlas.pkl', majority_voting = True,use_GPU=True)
# %%
predictions.predicted_labels

# %%
adata.obs = pd.concat([adata.obs, predictions.predicted_labels], axis = 1)

# # %%
# duplicates = predictions.predicted_labels.majority_voting.index.duplicated()
# print(predictions.predicted_labels.majority_voting.index[duplicates])
# predictions.predicted_labels.majority_voting = predictions.predicted_labels.majority_voting.loc[~duplicates]
# # %%
# adata_cows.obs["majority_voting_celltypist"] = predictions.predicted_labels.majority_voting

# %%
# plot the annotated cell types
sc.pl.umap(adata, color = 'majority_voting')#, legend_loc = 'on data')

# %%
sc.pl.umap(adata, color = 'predicted_labels')#, legend_loc = 'on data')


# %%
plt.figure(figsize = (20, 22))
sc.pl.umap(adata, color='predicted_labels', legend_loc='on data', legend_fontsize=2,frameon=False)
# %%
plt.figure(figsize = (20, 22))
sc.pl.umap(adata, color='majority_voting', legend_loc='on data', legend_fontsize=2,frameon=False)
# %%
sc.pl.umap(adata,color=['pct_counts_mt', 'pct_counts_ribo','pct_counts_hb', 'pct_counts_scov', ])
# %%
sc.pl.umap(adata,color=["LYVE1","ITGA2B","ACKR4","CD44","COCH","ANXA2","CD36","FABP4","ACKR3","BGN","FLRT2","IL33","MRC1","MARCO","PTX3","KCNJ8","ITIH5","CD209","S1PR1"])
# %%

# %%
sc.tl.rank_genes_groups(adata, groupby="leiden_scVI_0.1",  method="wilcoxon") #only work with this when labels are not provided
sc.pl.rank_genes_groups_dotplot(adata,n_genes=5)

# %%
sc.get.rank_genes_groups_df(adata, group="9")#.head(10)
# Check RGS6 (promoter of self renewal) https://pmc.ncbi.nlm.nih.gov/articles/PMC10709870/


# %%
sc.get.rank_genes_groups_df(adata, group="4")#.head(10)
#sc.get.rank_genes_groups_df(adata, group="4")["names"].head(5).tolist()
# mostly endothelial and angiogenesis genes


# %%
sc.tl.rank_genes_groups(adata, groupby="leiden_scVI_0.9",  method="wilcoxon") #only work with this when labels are not provided
sc.pl.rank_genes_groups_dotplot(adata,n_genes=3)

# %%
sc.pl.umap(adata,color=['leiden_scVI_0.9'],legend_loc="on data")

# %%
sc.get.rank_genes_groups_df(adata, group="9")#.head(10)

#INMT,PDE5A lung fibrosis



# %%
sc.get.rank_genes_groups_df(adata, group="2")#.head(10)

# %%
sc.get.rank_genes_groups_df(adata, group="20")#.head(10)

# %%
# 16 classicall monocytes
# 15,6, t cells
# 12 NK cells
# 21 , 10 B cells
# 20 renewing epithelial cells/capillary cells
# 2 vascular smooth  cells
# 5,0,8,7,3 endothetial and lymph vessels cells
# 13 fibrotic cells

# %%
plt.figure(figsize = (20, 22))
sc.pl.umap(adata,color="celltype_hint",legend_fontsize=2,legend_loc="on data",frameon=False)
# %%
adata.obs.celltype_hint.value_counts()

# %%
# Highligh the cell type STROMAL CELL that are present in the celltypist model
sc.pl.umap(adata,color="celltype_hint",groups=["stromal cell"],legend_loc="on data",legend_fontsize=2,frameon=False)

# %%
# Highlight the celltypes that are present in the celltypist model one by one in unique celltype_hint

for celltype in adata.obs.celltype_hint.unique():
    sc.pl.umap(adata,color="celltype_hint",groups=[celltype],legend_loc="on data",legend_fontsize=2,frameon=False)

# %%
# Rename some celltypes in the celltype_hint column
adata.obs
# %%
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["leiden_scVI_1.9"]
# %%
adata.obs.combined_celltype_hint_celltypist = adata.obs.combined_celltype_hint_celltypist.replace({
    "27":"Alveolar macrophages",
    '5':'Vascular smooth muscle',
    "20":'Vascular smooth muscle',
    '34':'Vascular smooth muscle',
    '35':'Vascular smooth muscle',
    '26':'Capillary endothelial cells',
    '0':'Endothelial cells',
    '1':'Inflammed Endothelial cells',
    '2':'Endothelial cells',
    '3':'Endothelial cells',
    '4':'Endothelial cells',
    '6':'Endothelial cells',
    '18':'Endothelial cells',
    '23':'Endothelial cells',
    '36':'Mast cells',
    '22':'Pericytes',
    '37':'Pericytes',
    '24':'Type II pneumocytes',
    '31': 'Platelets',
    '9':'Type I pneumocytes',
    '8':'Fibroblasts',
    '10':'B cells',
    '28':'Activated B cells',
    '21':'Neutrophils',

    #Improive the naming
    '30':'Dendritic cells',
    '11':'Inflammatory Macrophages',
    '7':'Inflammatory Macrophages',
    '17':'Monocytes/Macrophages',

    '14':'Macrophages',
    '15':'Monocytes',
    '16':'Monocytes',
    '32':'Monocytes',

    '30':'Dendritic cells',

    '29':'Progenitor cells',

    # T
    #13':'NK cells',
    #CD4 
    # CD8

    '12':'dummy',
    '19':'dummy',
    '25':'dummy',
    '33':'dummy',
    '13':'dummy',   


    })
# %%
sc.pl.umap(adata,color="combined_celltype_hint_celltypist",legend_loc="on data",legend_fontsize=4,frameon=False)





# %%
# Change the cluster names from Maste cells.
adata[adata.obs.majority_voting == "Classical monocytes"]
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Classical monocytes"])
adata.obs.loc[adata.obs.majority_voting == "Classical monocytes", "combined_celltype_hint_celltypist"] = "Classical monocytes"

adata[adata.obs.majority_voting == "Plasmacytoid DCs"]
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Plasmacytoid dendritic cells"])
adata.obs.loc[adata.obs.majority_voting == "Plasmacytoid DCs", "combined_celltype_hint_celltypist"] = "Plasmacytoid dendritic cells"


adata[adata.obs.majority_voting == 'Lymphatic EC differentiating']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Lymphatic EC differentiating"])
adata.obs.loc[adata.obs.majority_voting == 'Lymphatic EC differentiating', "combined_celltype_hint_celltypist"] = "Lymphatic EC differentiating"

adata[adata.obs.majority_voting == 'EC arterial']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["EC arterial"])
adata.obs.loc[adata.obs.majority_voting == 'EC arterial', "combined_celltype_hint_celltypist"] = "EC arterial"

adata[adata.obs.majority_voting == 'CD8 T cells']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["CD8 T cells"])
adata.obs.loc[adata.obs.majority_voting == 'CD8 T cells', "combined_celltype_hint_celltypist"] = "CD8 T cells"

adata[adata.obs.majority_voting == 'NK cells']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["NK cells"])
adata.obs.loc[adata.obs.majority_voting == 'NK cells', "combined_celltype_hint_celltypist"] = "NK cells"

adata[adata.obs.majority_voting == 'CD4 T cells']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["CD4 T cells"])
adata.obs.loc[adata.obs.majority_voting == 'CD4 T cells', "combined_celltype_hint_celltypist"] = "CD4 T cells"

adata[adata.obs.majority_voting == 'SMG mucous']
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["SMG mucous"])
adata.obs.loc[adata.obs.majority_voting == 'SMG mucous', "combined_celltype_hint_celltypist"] = "SMG mucous"


# %%
# Visualize the new celltypes
plt.figure(figsize = (20, 22))
sc.pl.umap(adata,color="combined_celltype_hint_celltypist",legend_loc="on data",legend_fontsize=6,frameon=False)

# %%
for celltype in adata.obs.combined_celltype_hint_celltypist.unique():
    sc.pl.umap(adata,color="combined_celltype_hint_celltypist",groups=[celltype],legend_loc="on data",legend_fontsize=4,frameon=False)

# %%
adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\05_Load_and_celltypist\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2024_12_17.h5ad")
# %%
