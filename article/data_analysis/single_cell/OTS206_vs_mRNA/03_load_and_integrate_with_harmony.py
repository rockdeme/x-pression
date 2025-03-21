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
from scipy.stats import median_abs_deviation



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
# Load the files from the filtered_feature_bc_matrix.h5 from the different samples folders
# base_path = "/data/users/mbotos/Environments/2024_11_29_Mouse_Lung_infected_Demeter/00_data/20230705_mouse_lung"
# samples = os.listdir(base_path)
# samples

base_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\02_Load_and_preprocess\adatas\filtered"
samples = os.listdir(base_path)
samples

adatas = []

for file in glob.glob(f"{base_path}/*.h5ad"):
    adata = sc.read_h5ad(file)
    print(f"Loaded {file}")
    adatas.append(adata)

print("Done loading files")

# %%
##########################################################################
# concatenate all the samples in one object for integration using scvi.  #
##########################################################################

print("Concatenating h5ad files...")
adatas = [adata for adata in adatas]
adata = anndata.concat(adatas, join="outer")
adata

# %%
print(adatas[1])

# %%
np.sum(adata.layers["raw_counts"],axis=1)
# %%
np.sum(adata.X,axis=1)
# # %%
# # sanity check
# adata.layers["raw_counts"] = adata.X.copy()
# np.max(adata.layers["raw_counts"])


# %%
# Make hard copy of raw object
adata.raw = adata
adata

# %%
# %%
# Get the number of available CPUs and GPUs from SLURM
#num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
num_cpus = os.cpu_count()
print(f"Number of CPUs available: {num_cpus}")
num_gpus = len(GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[]))
print(f"Number of GPUs available: {num_gpus}")

# %%
adata.obs["doublet_info"].value_counts()
# %%
# Filter out the doublets
print("Filtering out doublets...")
adata = adata[adata.obs["doublet_info"] == "singlet"].copy()
adata


# %%
# add metadata.csv file
metadata = pd.read_csv(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\metadata.csv")
metadata
#metadata.set_index("sample_id", inplace=True)
# %%
# Extract the last 6 rows
last_6_rows = metadata.tail(6)
# Ensure the 'sample_id' column in metadata matches the 'sample' column in adata.obs
if 'sample_id' in last_6_rows.columns:
    last_6_rows.rename(columns={'sample_id': 'sample'}, inplace=True)
# Merge the extracted metadata with adata.obs based on the 'sample' column
if 'sample' in adata.obs.columns and 'sample' in last_6_rows.columns:
    adata.obs = adata.obs.join(last_6_rows.set_index('sample'), on="sample", how="left")
adata.obs


# Check the merge h5ad does not save...

# %%
# Verify the merge
print(adata.obs['condition'].value_counts())
print(adata.obs['challenge'].value_counts())

# %%
# Add control and case metadata to the adata
print("Adding control and case metadata...")
adata.obs["replicate"] = adata.obs["sample"]#.str.split("_",expand=True)[0]
#adata.obs["sample"].str.extract(r'(....)')
#adata.obs["condition"] = adata.obs["replicate"].str.slice(0,1)
#adata.obs["replicate"].str.extract(r'([TC])')
#adata.obs["condition"] = adata.obs["replicate"].str.extract(r'(.)')
adata.obs



# %%
torch.set_float32_matmul_precision('high')
# %%

adata 
# %%
# print("Number of cells before filtering: ", adata.n_obs)
# # Filter genes
# sc.pp.filter_genes(adata, min_cells=50)
# # Filter cells
# sc.pp.filter_cells(adata, min_genes=300)
# print("Number of cells after filtering: ", adata.n_obs)

# Filters out in this case after a exhaustive check only 60 cells more but very harsh.

# %%
##################################################
#            RUN HARMONY INTEGRATION             #
##################################################
# Prepare the setup for harmony integration
print("Preparing Harmony integration...")
#scvi.model.SCVI.setup_anndata(adata, layer="raw_counts",batch_key="bio_batch")
#Add prepare basic scanpy processing to calculate PCA
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["log_counts"] = adata.X.copy()
#sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, n_comps=30)
sc.pl.pca_variance_ratio(adata, log=True)
sc.pl.pca_loadings(adata, components=range(1, 6), show=False)
sc.pl.pca(adata, color=["sample","condition"], components=["1,2","1,3","2,3"], show=False,ncols=3)
# %%
kwargs = {
     "max_iter_harmony": 20,  # Increase the number of iterations
    "epsilon_cluster": 1e-4,   # Adjust the convergence threshold
    "epsilon_harmony": 1e-4,# Convergence threshold for Harmony algorithm
    "plot_convergence": True,# Plot convergence
    #"return_object": True,  # Return full Harmony object
    "verbose": True         # Print progress messages

  }  
sc.external.pp.harmony_integrate(adata, key=["replicate"],**kwargs)#"condition","phase"]
# %%
# Define paths for saving the model and data
n_top_genes = "all"
# Construct the path with formatted variables
path = (rf"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\test_harmony\adatas_replicate_"
        f"{n_top_genes}")

# Create the directory if it doesn't exist
os.makedirs(path, exist_ok=True)
print(f"Directory created or exists: {path}")

#model_path = os.path.join(path, f"harmony_model_replicate_no_doublets_{n_top_genes}")
adata_path = os.path.join(path, f"harmony_model_replicate_no_doublets_{n_top_genes}.h5ad")

# Save the adata
print("Saving the adata...")
adata.write_h5ad(adata_path,compression="gzip")

# %%
# Run neighbors
print("Running neighbors...")
sc.pp.neighbors(adata, use_rep="X_pca_harmony",n_neighbors=15)#_corrected", n_neighbors=30, metric="cosine")
#sc.pp.neighbors(reference_latent, n_neighbors=30) the more neighbours the more "general" view of the data and less specific within the clusters
# %%
# Run leiden
print("Running leiden...")
# for resol in np.arange(0.1, 2, 0.2):
#     print("Resolution: " + str(round(resol, 3)))
#     sc.tl.leiden(reference_latent,resolution = resol, key_added = "leiden_scVI_"+str(round(resol,3)))

for resol in np.arange(0.1, 2, 0.2):
    print("Resolution: " + str(round(resol, 3)))
    sc.tl.leiden(adata,resolution = resol, key_added = "leiden_harmony_"+str(round(resol,3)))
# %%
print("Running UMAP...")
#sc.pp.neighbors(adata, use_rep="X_scVI")
#sc.tl.leiden(reference_latent,resolution = 0.5, key_added = "leiden_scVI_0.5")
sc.tl.umap(adata)#,min_dist=0.3,spread=4)
# # Save the adata
print("Saving the adata...")
adata.write_h5ad(adata_path, compression="gzip")

print("Done with best metrics")

# %%
adata.obs["phase_ordered"] = pd.Categorical(
    values=adata.obs.phase, categories=["G1","S","G2M"], ordered=True
)

# %%
path_umap = os.path.join(path, "figures", "umap")
os.makedirs(path_umap, exist_ok=True)
with plt.rc_context():  # Use this to set figure params like size and dpi
    sc.pl.umap(adata,color=["sample",
                            "condition",
                            "challenge",
                                 "n_genes_by_counts",
                                 "total_counts",
                                 "pct_counts_mt",
                                 "pct_counts_ribo",
                                 "pct_counts_hb",
                                 "pct_counts_scov",
                                 "log1p_n_genes_by_counts",
                                 "log1p_total_counts",
                                 

                                 "leiden_harmony_0.1",
                                 "leiden_harmony_0.3",
                                 "leiden_harmony_0.5",
                                 "leiden_harmony_0.7",
                                 "leiden_harmony_0.9",
                                 "leiden_harmony_1.1",
                                 "leiden_harmony_1.3",
                                 "leiden_harmony_1.5",

                                 "phase",
                                 "phase_ordered",
                                 "doublet_info",
                                 'Cd34','Mki67',#cell cycle active 
                                 'Aldh18a1',#synthetic mitochondria
                                 'Cd4','Cd8a','Cd5',#T cells
                                 'Flt3','Clec10a','Clec9a','Cadm1','Xcr1','Axl','Cd163', #cDC2,cDC1, tDC
                                 'Blnk','Tcf4','Ttyh1', # pDC
                                 'Fcgr3','Cd14', # Monocytes
                                 'Nkg7',# NK cells
                                 'Ms4a6c','Cd79a','Cd79b', # B cells
                                 'Csf3r','S100a8',#Neutrophils
                                 'ORF1ab', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N','ORF10',#SCOV genes
                                 'Hbb-bt', 'Hbb-bs', 'Hbb-bh2', 'Hbb-bh1', 'Hbb-y', 'Hbs1l', 'Hba-x',
                                 'Hba-a1', 'Hbq1b', 'Hba-a2', 'Hbq1a', 'Hbp1', 'Hbegf',#Hb genes
                                 ],
                                 ncols=2,vmin="p0.05",vmax="p99.5",
                                 frameon=False,show=False,layer="log_counts")
    plt.savefig(os.path.join(path_umap,f"_adata_harmony_normalized_{n_top_genes}_harmony_model_and_replicate.png"), bbox_inches="tight")
    plt.show()

# %%
# DE analysis
sc.tl.rank_genes_groups(adata,groupby='leiden_harmony_0.1',method='wilcoxon',key_added="wilcoxon")
sc.tl.rank_genes_groups(adata,groupby='leiden_harmony_0.1',method='t-test',key_added="t-test")
sc.tl.rank_genes_groups(adata,groupby='leiden_harmony_0.1',method='t-test_overestim_var',key_added="t-test_overestim_var")
sc.tl.rank_genes_groups(adata,groupby='leiden_harmony_0.1',method='logreg',key_added="logreg")


# %%
leiden_harmony_0_1_de_wc=adata.uns["wilcoxon"]['names']['0']
leiden_harmony_0_1_de_tt=adata.uns["t-test"]['names']['0']
leiden_harmony_0_1_de_ttestov=adata.uns["t-test_overestim_var"]['names']['0']
leiden_harmony_0_1_de_logreg=adata.uns["logreg"]['names']['0']


# %%
from matplotlib_venn import venn3

venn3([set(leiden_harmony_0_1_de_wc),set(leiden_harmony_0_1_de_tt),set(leiden_harmony_0_1_de_ttestov),],
      ('Wilcox','T-test','T-test_ov'))
plt.show()

# %%
venn3([set(leiden_harmony_0_1_de_wc),set(leiden_harmony_0_1_de_logreg),set(leiden_harmony_0_1_de_ttestov),],
      ('Wilcox','Logreg','T-test_ov'))

plt.show()

# %%
sc.pl.rank_genes_groups(adata, group_by="leiden_harmony_0.1",groups=["0"], n_genes=20, key="t-test",show_gene_labels=True,sharey=False)
sc.pl.rank_genes_groups(adata, group_by="leiden_harmony_0.1",groups=["0"], n_genes=20, key="logreg",show_gene_labels=True,sharey=False)
sc.pl.rank_genes_groups(adata, group_by="leiden_harmony_0.1",groups=["0"], n_genes=20, key="wilcoxon",show_gene_labels=True,sharey=False)
sc.pl.rank_genes_groups(adata, group_by="leiden_harmony_0.1",groups=["0"], n_genes=20, key="t-test_overestim_var",show_gene_labels=True,sharey=False)


# %%
adata.uns["t-test"]["names"]["0"]
adata.uns["t-test"]["pvals_adj"]["0"]
adata.uns["t-test"]["logfoldchanges"]["0"]
adata.uns["t-test"]["scores"]["0"]




# %%
leiden_harmony_0_1 = (
    adata.obs["leiden_harmony_0.1"]
    .value_counts()
    # .loc[lambda x: (x >= 500) & (x.index != "nan")]
    .loc[lambda x: x.index != "nan"]
    .to_frame("n_cells")
)
leiden_harmony_0_1.loc[:, "associated_test"] = leiden_harmony_0_1.index.astype(str) + " vs Rest"
leiden_harmony_0_1


# %%
# Make a dataframe with DE results and metadata, and add comparison for each cluster vs rest

for i in range(0,len(leiden_harmony_0_1)):
    print("Cluster ",i)
    leiden_harmony_0_1_de = pd.DataFrame(
        {
            "comparison": adata.uns["t-test"]["names"]["0"],
            "pval": adata.uns["t-test"]["pvals_adj"]["0"],
            "lfc_mean": adata.uns["t-test"]["logfoldchanges"]["0"],
            "lfc_std": adata.uns["t-test"]["scores"]["0"],
        }
    )
    # Add comparison column
    leiden_harmony_0_1_de["comparison"] = leiden_harmony_0_1.index[i] + " vs Rest"

    leiden_harmony_0_1_de["comparison"] = leiden_harmony_0_1_de["comparison"].astype(str)
    leiden_harmony_0_1_de

    leiden_harmony_0_1_de = leiden_harmony_0_1_de.join(adata.var, how="inner")
    leiden_harmony_0_1_de = leiden_harmony_0_1_de[leiden_harmony_0_1_de[["scale1", "scale2"]].max(axis=1) > 1e-4]
    leiden_harmony_0_1_de.head(20)

### CONTINUE HERE















# %%
# Extract top 20 DE genes
# This cell extracts list of top 5 upregulated genes for every cell-type
marker_genes = (
    leiden_harmony_0_1_de.reset_index()
    .loc[lambda x: x.comparison.isin(leiden_harmony_0_1.associated_test.values)]
    .groupby("comparison")
    .apply(
        lambda x: x.sort_values("lfc_mean", ascending=False).iloc[:20]
    )  # Select top 5 DE genes per comparison
    .reset_index(drop=True)#[["feature_name", "soma_joinid"]]
    .drop_duplicates()
)
# %%
marker_genes

# %%
marker_genes.to_csv(os.path.join(path, f"marker_genes_leiden_harmony_0.1_{n_top_genes}.csv"), index=False)

# %%
for i in range(0,len(marker_genes["comparison"].unique())):
    print("Cluster ",i)
    top_genes_cluster = marker_genes[marker_genes["comparison"] == marker_genes["comparison"].unique()[i]]["index"].tolist()#values.tolist()
    sc.pl.umap(adata, color=top_genes_cluster + ["leiden_scVI_0.1"], ncols=2, frameon=False, show=False,layer="scvi_normalized",vmin="p0.05",vmax="p99.5")


# %%