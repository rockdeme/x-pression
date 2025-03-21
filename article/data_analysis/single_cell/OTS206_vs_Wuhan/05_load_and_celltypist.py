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
adata = sc.read_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\04_Load_and_scimilarity\adatas\adata_preprocessed_integrated_scVI_annotated_scimilarity.h5ad")


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
#model = models.Model.load(model = "Autopsy_COVID19_Lung.pkl")
#model = models.Model.load(model = "Cells_Lung_Airway.pkl")
print(model)
model.cell_types

# %%
adata
# # %%
# # Some cells were duplicated for some reason in the adata.
# adata_duplicated_cells = adata.obs.index.duplicated(keep = False)
# adata_duplicated_cells
# # %%
# adata.obs['adata_duplicated_cells'] = np.where(adata_duplicated_cells, 'duplicated', 'non-duplicated')
# sc.pl.umap(adata, color = 'adata_duplicated_cells')
# # %%
# adata

# # %%
# # Remove duplicated cells
# non_duplicates = ~adata.obs.index.duplicated(keep='first')
# adata = adata[non_duplicates].copy()  
# adata = adata[~adata.obs['adata_duplicated_cells']].copy()
# %%
np.max(adata.X)
# %%
np.max(adata.layers["raw_counts"])
# %%
adata.X = adata.layers["raw_counts"]
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
adata.layers["norm_counts"] = adata.X.copy()
adata.layers["log_counts"] = np.log1p(adata.layers["norm_counts"])
adata.X = adata.layers["log_counts"]
#sc.pp.log1p(adata)

# %%
predictions = celltypist.annotate(adata, model = 'Human_Lung_Atlas.pkl', majority_voting = True,use_GPU=True)
# %%
predictions.predicted_labels

# %%
# drop predictions from adata, comment if it is the first time you are running this cell
# adata.obs.drop(columns = "predicted_labels", inplace = True)
# adata.obs.drop(columns = "majority_voting", inplace = True)

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
#######################
# Visualize the data  #
#######################

sc.pl.umap(adata,color=['pct_counts_mt', 'pct_counts_ribo','pct_counts_hb', 'pct_counts_scov', ],ncols=2)
# %%
sc.pl.umap(adata,color=["LYVE1","ITGA2B","ACKR4","CD44","COCH","ANXA2","CD36","FABP4","ACKR3","BGN","FLRT2","IL33","MRC1","MARCO","PTX3","KCNJ8","ITIH5","CD209","S1PR1"])
# %%

# %%
sc.tl.rank_genes_groups(adata, groupby="leiden_scVI_1.9",  method="wilcoxon") #only work with this when labels are not provided
sc.pl.rank_genes_groups_dotplot(adata,n_genes=5)

# %%
sc.pl.umap(adata,color=['leiden_scVI_1.9'],legend_loc="on data")

# %%
sc.get.rank_genes_groups_df(adata, group="14")#.head(10)
# LARS2, SCGB1A1
# %%
sc.pl.umap(adata,color=["LARS2","SCGB1A1"],ncols=2)


# %%
sc.get.rank_genes_groups_df(adata, group="15")#.head(10)



# %%
#INMT,PDE5A lung fibrosis

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
sc.pl.umap(adata,color="celltype_hint",legend_fontsize=4,legend_loc="on data",frameon=False)
# %%
adata.obs.celltype_hint.value_counts()

# %%
# Highligh the cell type STROMAL CELL that are present in the celltypist model
sc.pl.umap(adata,color="celltype_hint",groups=["stromal cell"],legend_loc="on data",legend_fontsize=4,frameon=False)

# %%
# Highlight the celltypes that are present in the celltypist model one by one in unique celltype_hint
###################
#For scimilarity  #
###################
for celltype in adata.obs.celltype_hint.unique():
    sc.pl.umap(adata,color="celltype_hint",groups=[celltype],legend_loc="on data",legend_fontsize=8,frameon=False)


# %%
##################### 
# for celltypist    #
#####################
#gene_set="Human_Lung_Atlas.pkl"
#gene_set="Autopsy_COVID19_Lung.pkl"
#gene_set="Cells_Lung_Airway.pkl"
#gene_set="Lethal_COVID19_Lung.pkl"
gene_set="Human_PF_Lung.pkl"


model = models.Model.load(model = gene_set)
print(model)
model.cell_types
adata.X = adata.layers["raw_counts"].copy()
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
adata.layers["norm_counts"] = adata.X.copy()
adata.layers["log_counts"] = np.log1p(adata.layers["norm_counts"])
adata.X = adata.layers["log_counts"]
#sc.pp.log1p(adata)
predictions = celltypist.annotate(adata, model = gene_set, majority_voting = True,use_GPU=True)
predictions.predicted_labels

# drop predictions from adata, comment if it is the first time you are running this cell
adata.obs.drop(columns = "predicted_labels", inplace = True)
adata.obs.drop(columns = "majority_voting", inplace = True)
adata.obs = pd.concat([adata.obs, predictions.predicted_labels], axis = 1)

for celltype in adata.obs.predicted_labels.unique():
    sc.pl.umap(adata,color="predicted_labels",groups=[celltype],legend_loc="on data",legend_fontsize=8,frameon=False)

# %%
for celltype in adata.obs.predicted_labels.unique():
    sc.pl.embedding(adata,basis="X_umap_X_scVI_latent_umap", color="predicted_labels",groups=[celltype],
                    legend_loc="on data",legend_fontsize=8,frameon=False)

# %% Check the other results from scimilarity
for celltype in adata.obs.predictions_unconstrained.unique():
    sc.pl.embedding(adata,basis="X_umap", color="predictions_unconstrained",groups=[celltype],
                    legend_loc="on data",legend_fontsize=8,frameon=False)

# %%
# Plot the Myofibroblasts cells in the celltypist model
#celltype_to_highlight = "Myofibroblasts"
#celltype_to_highlight = "Lymphatic EC differentiating"
#celltype_to_highlight = "Lymphatic Endothelial Cells"
#celltype_to_highlight = "PLIN2+ Fibroblasts"
#celltype_to_highlight = "Endothelial Cells"
celltype_to_highlight = "epithelial cell"


sc.pl.umap(adata,color="predicted_labels",groups=[celltype_to_highlight],legend_loc="on data",legend_fontsize=4,frameon=False,palette="tab20")

# %%
sc.pl.umap(adata,color="celltype_hint",groups=[celltype_to_highlight],legend_loc="on data",legend_fontsize=4,frameon=False,palette="tab20")


# %%
adata.obs

# %% Check the celltypes in the UMAP alternative to scanpy from scimilarity visualization
plt.figure(figsize=(40, 24))
sc.pl.embedding(adata,basis="X_umap_X_scVI_latent_umap", color=["celltype_hint","leiden_scVI_1.9"],legend_loc="on data",legend_fontsize=4,frameon=False,ncols=1,show=False)
plt.show()



# %%
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["leiden_scVI_1.9"]
sc.pl.umap(adata, color="combined_celltype_hint_celltypist",legend_loc="on data",legend_fontsize=8,frameon=False)

# %%
# LARS2 is in 14 too with COVID
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["leiden_scVI_1.9"]
adata.obs.combined_celltype_hint_celltypist = adata.obs.combined_celltype_hint_celltypist.replace({
    "0":"Endothelial cells",
    "1": "B ccells",
    "2": "Endothelial cells",
    "3": "Endothelial cells",
    "4": "Monocytes",
    "5": "Endothelial cells",
    "6": "T cells",
    "7": "Neutrophils",
    "8": "EC aerocyte ",
    "9": "NK Cells",
    "10": "Endothelial cells",
    "11": "Monocytes",
    "12": "Endothelial cells",
    "13": "Endothelial cells",
    "14": "SCOV2 AT2 cells",
    "15": "Progenitor cells",
    "16": "AT1 cells",
    "17": "endotheleal cells of lymphatic vessel",
    "18": "Endothelial cells",
    "19": "Fibroblasts",
    "20": "Macrophages",
    "21": "NK T cells",
    "22": "Fibroblasts",
    "23": "EC aerocyte",
    "24": "Smooth muscle cells",
    "25": "Dendritic cells",
    "26": "Smooth muscle cells",
    "27": "Mast cells",
    })
# OLD CELL ANNOTATIONS from E1 to E6
#adata.obs.combined_celltype_hint_celltypist = adata.obs.combined_celltype_hint_celltypist.replace({
    # "27":"Alveolar macrophages",
    # '5':'Vascular smooth muscle',
    # "20":'Vascular smooth muscle',
    # '34':'Vascular smooth muscle',
    # '35':'Vascular smooth muscle',
    # '26':'Capillary endothelial cells',
    # '0':'Endothelial cells',
    # '1':'Inflammed Endothelial cells',
    # '2':'Endothelial cells',
    # '3':'Endothelial cells',
    # '4':'Endothelial cells',
    # '6':'Endothelial cells',
    # '18':'Endothelial cells',
    # '23':'Endothelial cells',
    # '36':'Mast cells',
    # '22':'Pericytes',
    # '37':'Pericytes',
    # '24':'Type II pneumocytes',
    # '31': 'Platelets',
    # '9':'Type I pneumocytes',
    # '8':'Fibroblasts',
    # '10':'B cells',
    # '28':'Activated B cells',
    # '21':'Neutrophils',

    # #Improive the naming
    # '30':'Dendritic cells',
    # '11':'Inflammatory Macrophages',
    # '7':'Inflammatory Macrophages',
    # '17':'Monocytes/Macrophages',

    # '14':'Macrophages',
    # '15':'Monocytes',
    # '16':'Monocytes',
    # '32':'Monocytes',

    # '30':'Dendritic cells',

    # '29':'Progenitor cells',

    # # T
    # #13':'NK cells',
    # #CD4 
    # # CD8

    # '12':'dummy',
    # '19':'dummy',
    # '25':'dummy',
    # '33':'dummy',
    # '13':'dummy',   
    # })
# %%
sc.pl.umap(adata,color="combined_celltype_hint_celltypist",legend_loc="on data",legend_fontsize=4,frameon=False)

# %% Visu celltype by celltype

for celltype in adata.obs.combined_celltype_hint_celltypist.unique():
    sc.pl.umap(adata,color="combined_celltype_hint_celltypist",groups=[celltype],legend_loc="on data",legend_fontsize=4,frameon=False)


# %%
# Clean the cells from cluster 8 that are very appart
import matplotlib.pyplot as plt

# Select cells in cluster "8"
cluster8_mask = adata.obs["leiden_scVI_1.9"] == "8"
umap_cluster8 = adata.obsm["X_umap"][cluster8_mask]

plt.figure(figsize=(6,5))
plt.scatter(umap_cluster8[:, 0], umap_cluster8[:, 1], s=10, alpha=0.7)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Cluster 8")
plt.show()

# %%
# Define the condition based on UMAP coordinates
mask_to_reassign = (adata.obs["leiden_scVI_1.9"] == "8") & \
                   (adata.obsm["X_umap"][:, 0] < 9.0) & \
                   (adata.obsm["X_umap"][:, 1] < 4.0)

print("Number of cells to reassign:", mask_to_reassign.sum())

# %%
# Clean the NK T cells cluster
# This was done after annotating as I saw some cells that were not NK T cells

cluster_nkt_mask = adata.obs["combined_celltype_hint_celltypist"] == "NK T cells"
umap_cluster_nkt = adata.obsm["X_umap"][cluster_nkt_mask]

plt.figure(figsize=(6,5))
plt.scatter(umap_cluster_nkt[:, 0], umap_cluster_nkt[:, 1], s=10, alpha=0.7)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Cluster NK T cells")
plt.show()
# %%
# Define the condition based on UMAP coordinates
mask_to_reassign_2 = (adata.obs["combined_celltype_hint_celltypist"] == "NK T cells") & \
                   (adata.obsm["X_umap"][:, 0] > 12.0) & \
                   (adata.obsm["X_umap"][:, 1] > 4.0)

print("Number of cells to reassign:", mask_to_reassign_2.sum())




# %%
# Reassign the cluster
#adata.obs.loc[mask_to_reassign, "leiden_scVI_1.9"] = "28"
# Add the new category "28" to the categorical column
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["leiden_scVI_1.9"]
#adata.obs["leiden_scVI_1.9"] = adata.obs["leiden_scVI_1.9"].cat.add_categories(["28"])
#adata.obs["leiden_scVI_1.9"] = adata.obs["leiden_scVI_1.9"].cat.remove_categories(["28"])


# Now reassign the cells matching your mask to Monocytes, 11
adata.obs.loc[mask_to_reassign, "leiden_scVI_1.9"] = "11"
adata.obs["combined_celltype_hint_celltypist"] = adata.obs["leiden_scVI_1.9"]

# Reassign the cluster of NK T cells, to 13
adata.obs.loc[mask_to_reassign_2, "combined_celltype_hint_celltypist"] = "13"


# %% Reassing names and check the new celltypes
adata.obs.combined_celltype_hint_celltypist = adata.obs.combined_celltype_hint_celltypist.replace({
    "0":"Endothelial cells",
    "1": "B ccells",
    "2": "Endothelial cells",
    "3": "Endothelial cells",
    "4": "Monocytes",
    "5": "Endothelial cells",
    "6": "T cells",
    "7": "Neutrophils",
    "8": "EC aerocyte ",
    "9": "NK Cells",
    "10": "Endothelial cells",
    "11": "Monocytes",
    "12": "Endothelial cells",
    "13": "Endothelial cells",
    "14": "SCOV2 AT2 cells",
    "15": "Progenitor cells",
    "16": "AT1 cells",
    "17": "endotheleal cells of lymphatic vessel",
    "18": "Endothelial cells",
    "19": "Fibroblasts",
    "20": "Macrophages",
    "21": "NK T cells",
    "22": "Fibroblasts",
    "23": "EC aerocyte",
    "24": "Smooth muscle cells",
    "25": "Dendritic cells",
    "26": "Smooth muscle cells",
    "27": "Mast cells",
    })




# %%
# # %%
# # # Change the cluster names from Maste cells.
# adata[adata.obs.majority_voting == "Classical monocytes"]
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Classical monocytes"])
# adata.obs.loc[adata.obs.majority_voting == "Classical monocytes", "combined_celltype_hint_celltypist"] = "Classical monocytes"

# adata[adata.obs.majority_voting == "Plasmacytoid DCs"]
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Plasmacytoid dendritic cells"])
# adata.obs.loc[adata.obs.majority_voting == "Plasmacytoid DCs", "combined_celltype_hint_celltypist"] = "Plasmacytoid dendritic cells"


# adata[adata.obs.majority_voting == 'Lymphatic EC differentiating']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["Lymphatic EC differentiating"])
# adata.obs.loc[adata.obs.majority_voting == 'Lymphatic EC differentiating', "combined_celltype_hint_celltypist"] = "Lymphatic EC differentiating"

# adata[adata.obs.majority_voting == 'EC arterial']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["EC arterial"])
# adata.obs.loc[adata.obs.majority_voting == 'EC arterial', "combined_celltype_hint_celltypist"] = "EC arterial"

# adata[adata.obs.majority_voting == 'CD8 T cells']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["CD8 T cells"])
# adata.obs.loc[adata.obs.majority_voting == 'CD8 T cells', "combined_celltype_hint_celltypist"] = "CD8 T cells"

# adata[adata.obs.majority_voting == 'NK cells']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["NK cells"])
# adata.obs.loc[adata.obs.majority_voting == 'NK cells', "combined_celltype_hint_celltypist"] = "NK cells"

# adata[adata.obs.majority_voting == 'CD4 T cells']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["CD4 T cells"])
# adata.obs.loc[adata.obs.majority_voting == 'CD4 T cells', "combined_celltype_hint_celltypist"] = "CD4 T cells"

# adata[adata.obs.majority_voting == 'SMG mucous']
# adata.obs["combined_celltype_hint_celltypist"] = adata.obs["combined_celltype_hint_celltypist"].cat.add_categories(["SMG mucous"])
# adata.obs.loc[adata.obs.majority_voting == 'SMG mucous', "combined_celltype_hint_celltypist"] = "SMG mucous"


# %%
# Visualize the new celltypes
plt.figure(figsize = (20, 22))
sc.pl.umap(adata,color="combined_celltype_hint_celltypist",legend_loc="on data",legend_fontsize=6,frameon=False)

# %% celltype by celltype
for celltype in adata.obs.combined_celltype_hint_celltypist.unique():
    sc.pl.umap(adata,color="combined_celltype_hint_celltypist",groups=[celltype],legend_loc="on data",legend_fontsize=4,frameon=False)



# %%
# The automatic annotation is not perfect, so I tried to assign the celltypes manually
sc.pl.umap(adata,color="celltype_hint",legend_loc="on data",legend_fontsize=4,frameon=False)#groups=[celltype],

# %% Save the object to a file
# Use the combined_celltype_hint_celltypist or the celltype_hint if preffer to use the standard annotation from the celltypist model

print("use the column celltype_hint for the celltype annotation of the package celltypist")
print("use the column combined_celltype_hint_celltypist for the automatic and manual celltype annotation")

adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\05_Load_and_celltypist\adata_preprocessed_integrated_scVI_annotated_scimilarity_celltypist_2025_02_21.h5ad")
# %%
print("Done"*10)
# %%
