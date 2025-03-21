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

base_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\02_Load_and_preprocess\adatas\filtered"
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
# Extract the samples from the metadata
print("Extracting metadata...")
metadata.sample_id[metadata.sample_id.str.contains(r"pilot_",regex=True)]
#metadata.sample_id = metadata.sample_id.str.replace(r"pilot_", "")
#metadata.loc[metadata.sample_id.str.contains(r"pilot_",regex=True),"sample_id"] = metadata.loc[metadata.sample_id.str.contains(r"pilot_",regex=True),"sample_id"].str.replace(r"pilot_", "")
metadata_pilot = metadata.loc[metadata.sample_id.str.contains(r"pilot_",regex=True)]
metadata_pilot
#last_6_rows = metadata.tail(6)
# Ensure the 'sample_id' column in metadata matches the 'sample' column in adata.obs
if 'sample_id' in metadata_pilot.columns:
    metadata_pilot.rename(columns={'sample_id': 'sample'}, inplace=True)
# Merge the extracted metadata with adata.obs based on the 'sample' column
if 'sample' in adata.obs.columns and 'sample' in metadata_pilot.columns:
    adata.obs = adata.obs.join(metadata_pilot.set_index('sample'), on="sample", how="left")
adata.obs


# Check the merge h5ad does not save...

# %%
# Verify the merge
print(adata.obs['condition'].value_counts())
print(adata.obs['challenge'].value_counts())
print(adata.obs['timepoint'].value_counts())

# %%
# Add control and case metadata to the adata
print("Adding control and case metadata...")
adata.obs["replicate"] = adata.obs["sample"]#.str.split("_",expand=True)[0]
#adata.obs["sample"].str.extract(r'(....)')
#adata.obs["condition"] = adata.obs["replicate"].str.slice(0,1)
#adata.obs["replicate"].str.extract(r'([TC])')
#adata.obs["condition"] = adata.obs["replicate"].str.extract(r'(.)')
adata.obs

#######################################
# Inspect just with a merge the data  #
#######################################
# IF samples are good will overlap
# %%
adata_copy = adata.copy() #Create a copy of the data to work with
adata_copy.X = adata_copy.layers["raw_counts"].copy()
#Normalize
sc.pp.normalize_total(adata_copy, target_sum=1e4)
adata_copy.layers["normalized_1e4"] = adata_copy.X.copy()
#Log
#sc.pp.log1p(adata_copy)
#adata_copy.layers["logcounts"] = adata_copy.X.copy()
adata_copy.layers["logcounts"] = np.log1p(adata_copy.layers["normalized_1e4"])
sc.pp.highly_variable_genes(adata_copy,layer="logcounts",flavor="seurat")
                            # min_mean=0.0125,
                            # max_mean=3,
                            # min_disp=0.5)#,subset=True)
print("Running PCA for sample: " + str(adata_copy.obs["sample"].iloc[0]))
sc.tl.pca(adata_copy)
sc.pp.neighbors(adata_copy,n_pcs=30,n_neighbors=15)
print("Running UMAP for sample: " + str(adata_copy.obs["sample"].iloc[0]))
sc.tl.umap(adata_copy)
for i in np.arange(0,1,0.1):
    sc.tl.leiden(adata_copy,resolution=i,key_added="leiden_scVI_"+str(round(i,3)))
print("Plotting UMAP for sample: " + str(adata_copy.obs["sample"].iloc[0]))
sc.pl.umap(adata_copy,color=['sample','species', 'tissue',
                             'n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'log1p_n_genes_by_counts',
                             "pct_counts_mt","pct_counts_hb","pct_counts_ribo","pct_counts_scov",
                             #'doublet_info',
                             'S_score', 'G2M_score', 'phase',
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
                             "leiden_scVI_0.1","leiden_scVI_0.2","leiden_scVI_0.3","leiden_scVI_0.5","leiden_scVI_0.7","leiden_scVI_0.9"],
                             layer="logcounts",
                             ncols=3)#'batch', 'celltypes', 'tissue', 'method', 'Species', 'Method', 'Method_version'
print("Quality checking of the raw datasets complete.")

# they look awesome with only merge...
# %%
# Determine the batch_size of the dataset...
adata_copy.obs["leiden_scVI_0.9"].value_counts()


# %%

##############################
#    scVI integration        #
##############################
# %%
torch.set_float32_matmul_precision('high')
# %%

adata 
# %%
# Prepare the setup for scvi integration

best_hyperparameters = {'n_hidden':64,
                        'n_layers': 2,
                        'n_latent': 30,
                        'batch_size': 64,
                        #'lr': 0.000884204971440018,
                        "lr": 1e-3,
                        'dropout_rate': 0.1,
                        #'use_batch_norm': 'decoder',
                        #'use_layer_norm': 'encoder',
                        #'encode_covariates': False,
                        #'deeply_inject_covariates': False
                        }

# Check if CUDA is available and set the devices
if torch.cuda.is_available():
    print("CUDA is available")
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available")

# Load your actual data

# Check if 'raw_counts' is a valid key
if 'raw_counts' not in adata.layers:
    scvi.model.SCVI.setup_anndata(adata, batch_key="replicate")
else:
    # scvi.model.SCVI.setup_anndata(adata, layer="raw_counts",batch_key="bio_batch")
    scvi.model.SCVI.setup_anndata(adata, layer="raw_counts",
                                  batch_key="replicate",)
                                  #categorical_covariate_keys=["replicate"])
                            #   continuous_covariate_keys= ["log1p_n_genes_by_counts",
                            #                               "log1p_total_counts",
                            #                               "pct_counts_mt",
                            #                               "pct_counts_ribo",
                            #                               "pct_counts_hb"])
                         

# %%
print("Using as batch: ",adata.obs.replicate.unique())
# Define the model parameters using model_kwargs
model_kwargs = {
    "dropout_rate": best_hyperparameters['dropout_rate'],
    #"encode_covariates": best_hyperparameters['encode_covariates'],
    #"deeply_inject_covariates": best_hyperparameters['deeply_inject_covariates'],
    #"use_layer_norm": best_hyperparameters['use_layer_norm'],
    #"use_batch_norm": best_hyperparameters['use_batch_norm'],
    "n_hidden": best_hyperparameters['n_hidden'],
    "n_layers": best_hyperparameters['n_layers'],
    "n_latent": best_hyperparameters['n_latent'],
}

# Initialize the SCVI model with custom parameters
model = scvi.model.SCVI(adata=adata, **model_kwargs)

# Set the number of workers and pin memory
# scvi.settings.dl_num_workers = num_cpus - 1
# scvi.settings.dl_pin_memory = True
# print(model)
print("Model setup done, ready for training...")
model

# %%
# Define the training parameters using train_kwargs
train_kwargs = {
    "max_epochs": 400,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_min_delta": 0.00,
    "early_stopping_patience": 10,
    "early_stopping_mode": "min",
    "accelerator": 'gpu',  # Use GPU accelerator
    #"devices": -1,  # Use the number of available CUDA devices, interactive session does not allow it, so it cracks with -1
    "batch_size": best_hyperparameters['batch_size'],
    #"num_workers": num_cpus -1  # Use the number of CPUs allocated by SLURM, minus one for the main process
    
}

# Define the plan parameters using plan_kwargs for learning rate
plan_kwargs={"lr": best_hyperparameters['lr']}

# Train the model
print("Model setup done, ready for training...")
print("Training the model...")
model.train(**train_kwargs, plan_kwargs=plan_kwargs)


# %%
# Get the latent space
print("Getting the latent space...")
adata.obsm['X_scVI_latent'] = model.get_latent_representation()
adata.layers["scvi_normalized"] = model.get_normalized_expression()

# Obtain the decoder
decoder = model.module.decoder
print("Decoder obtained:", decoder)


# Define paths for saving the model and data
n_top_genes = "all"
# Construct the path with formatted variables
path = (
    rf"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\Mock_OTS_WT\03_Load_and_integrate\scVI\adatas\scvi_model_replicate_no_doublets_"
    f"{best_hyperparameters['dropout_rate']}_{best_hyperparameters['n_hidden']}_{best_hyperparameters['n_layers']}_{best_hyperparameters['n_latent']}_{best_hyperparameters['batch_size']}_{n_top_genes}"
)

# Create the directory if it doesn't exist
os.makedirs(path, exist_ok=True)
print(f"Directory created or exists: {path}")

model_path = os.path.join(path, f"scvi_model_replicate_no_doublets_{best_hyperparameters['dropout_rate']}_{best_hyperparameters['n_hidden']}_{best_hyperparameters['n_layers']}_{best_hyperparameters['n_latent']}_{best_hyperparameters['batch_size']}_{n_top_genes}")
adata_path = os.path.join(path, f"scvi_model_replicate_no_doublets_{best_hyperparameters['dropout_rate']}_{best_hyperparameters['n_hidden']}_{best_hyperparameters['n_layers']}_{best_hyperparameters['n_latent']}_{best_hyperparameters['batch_size']}_{n_top_genes}.h5ad")
training_curve_path = os.path.join(path, f"training_curve_scvi_model___replicate_{n_top_genes}_no_doublets.png")
training_loss_path = os.path.join(path, f"training_loss_scvi_model___replicate_{n_top_genes}_no_doublets.png")
kl_divergence_path = os.path.join(path, f"kl_divergenece_scvi_model___replicate_{n_top_genes}_no_doublets.png")


# Save the model
print("Saving the model...")
model.save(model_path)

# Save the adata
print("Saving the adata...")
adata.write_h5ad(adata_path,compression="gzip")

# Function to plot and save figures
def plot_and_save(history_key, title, ylabel, save_path, xlabel="Epochs"):
    plt.figure(figsize=(12, 6))
    plt.plot(model.history[history_key], label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()

# Plot and save the ELBO values
print("Plotting the ELBO values...")
plot_and_save("elbo_train", "Training ELBO", "ELBO", training_curve_path)
plot_and_save("elbo_validation", "Validation ELBO", "ELBO", training_curve_path)

# Plot and save the training loss per step
print("Plotting the training loss per step...")
plot_and_save("train_loss_step", "Training Loss per Step", "Loss", training_loss_path)

# Ensure convergence
print("Plotting the training and validation ELBO values...")
train_test_results = model.history["elbo_train"]
train_test_results["elbo_validation"] = model.history["elbo_validation"]
train_test_results.iloc[10:].plot(logy=True)  # exclude first 10 epochs

# Save the plot
convergence_plot_path = os.path.join(path, f"convergence_plot_scvi_model___replicate_{n_top_genes}_no_doublets.png")
plt.savefig(convergence_plot_path)
plt.show()
plt.close()

# Plot and save the KL divergence if available
if "kl_local_train" in model.history:
    print("Plotting the KL divergence...")
    plot_and_save("kl_local_train", "Training KL Divergence", "KL Divergence", kl_divergence_path)
# %%
# Run neighbors
print("Running neighbors...")
sc.pp.neighbors(adata, use_rep="X_scVI_latent",n_neighbors=15)#_corrected", n_neighbors=30, metric="cosine")
#sc.pp.neighbors(reference_latent, n_neighbors=30) the more neighbours the more "general" view of the data and less specific within the clusters
# %%
# Run leiden
print("Running leiden...")
# for resol in np.arange(0.1, 2, 0.2):
#     print("Resolution: " + str(round(resol, 3)))
#     sc.tl.leiden(reference_latent,resolution = resol, key_added = "leiden_scVI_"+str(round(resol,3)))

for resol in np.arange(0.1, 2, 0.2):
    print("Resolution: " + str(round(resol, 3)))
    sc.tl.leiden(adata,resolution = resol, key_added = "leiden_scVI_"+str(round(resol,3)))
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
                                 

                                 "leiden_scVI_0.1",
                                 "leiden_scVI_0.3",
                                 "leiden_scVI_0.5",
                                 "leiden_scVI_0.7",
                                 "leiden_scVI_0.9",
                                 "leiden_scVI_1.1",
                                 "leiden_scVI_1.3",
                                 "leiden_scVI_1.5",

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
                                 frameon=False,show=False,layer="scvi_normalized")
    plt.savefig(os.path.join(path_umap,f"_adata_scvi_normalized_{n_top_genes}_scvi_model_and_replicate.png"), bbox_inches="tight")
    plt.show()

# %%
adata.X = adata.layers["raw_counts"].copy()
#Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
adata.layers["normalized_1e4"] = adata.X.copy()
#Log
#sc.pp.log1p(adata)
adata.layers["log_counts"] = np.log1p(adata.layers["normalized_1e4"])


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a Seurat-like colormap with more dynamic range
seurat_cmap = LinearSegmentedColormap.from_list(
    "seurat_cmap", ["#dddddd", "#5603f5"], N=512  # Increase N for better gradient resolution
)

# Apply gamma correction to enhance contrast
def adjust_colormap(cmap, gamma=1.5):
    colors = cmap(np.linspace(0, 1, 256))  # Sample original colormap
    colors[:, :3] = colors[:, :3] ** gamma  # Apply gamma correction to RGB channels
    return LinearSegmentedColormap.from_list("adjusted_cmap", colors)

# Adjusted colormap with enhanced contrast
seurat_cmap_adjusted = adjust_colormap(seurat_cmap, gamma=1.8)

# Visualize the colormap
fig, ax = plt.subplots(figsize=(8, 1))
plt.colorbar(plt.cm.ScalarMappable(cmap=seurat_cmap_adjusted), cax=ax, orientation="horizontal")
plt.title("Enhanced Seurat-like Colormap")
plt.show()


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
                                 

                                 "leiden_scVI_0.1",
                                 "leiden_scVI_0.3",
                                 "leiden_scVI_0.5",
                                 "leiden_scVI_0.7",
                                 "leiden_scVI_0.9",
                                 "leiden_scVI_1.1",
                                 "leiden_scVI_1.3",
                                 "leiden_scVI_1.5",

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
                                 frameon=False,show=False,layer="log_counts",
                                 cmap=seurat_cmap_adjusted)
    plt.savefig(os.path.join(path_umap,f"_adata_log_counts_{n_top_genes}_scvi_model_and_replicate.png"), bbox_inches="tight")
    plt.show()


# %%
# Plot like Seurat
for gene in ["S", "M", "Cd34", "Flt3", "Cd14", "Nkg7", "Cd79a", "Csf3r", "ORF1ab","E","Hbb-bt"]:
    gene_expression = adata[:, gene].layers["log_counts"].toarray().flatten()
    non_zero_expression = gene_expression[gene_expression > 0]

    p10 = np.percentile(non_zero_expression, 10)
    p90 = np.percentile(non_zero_expression, 90)

    sc.pl.umap(
        adata,
        color=gene,
        cmap=seurat_cmap_adjusted,     # Light grey to blue gradient
        size=55,              # Adjust to match Seurat's dot size
        alpha=0.75,            # Transparency for blending effect
        frameon=False,         # Removes box around the plot (like Seurat)
        vmin=p10,vmax=p90,
        layer="log_counts",
        show=False,
        colorbar_loc=None,
    )
    plt.title(gene, fontsize=16)
    #plt.savefig(f"{gene}_seuratisezd_non_legend.svg", dpi=300)
    plt.savefig(os.path.join(path_umap,f"{gene}_seuratisezd_without_legend.svg"), dpi=300)
    # add the title size larger
    plt.show()
    plt.close()



# %%
print("Done\n"*10)

print("Best parameters so far here are the best hyperparameters:",best_hyperparameters, "left right now witht the path:",path)










# %%
# DE analysis
leiden_scVI_0_1_de = model.differential_expression(adata,groupby='leiden_scVI_0.1')
leiden_scVI_0_1_de

# %%
leiden_scVI_0_1 = (
    adata.obs["leiden_scVI_0.1"]
    .value_counts()
    # .loc[lambda x: (x >= 500) & (x.index != "nan")]
    .loc[lambda x: x.index != "nan"]
    .to_frame("n_cells")
)
leiden_scVI_0_1.loc[:, "associated_test"] = leiden_scVI_0_1.index.astype(str) + " vs Rest"
leiden_scVI_0_1


# %%
leiden_scVI_0_1_de = leiden_scVI_0_1_de.join(model.adata.var, how="inner")
leiden_scVI_0_1_de = leiden_scVI_0_1_de[
    leiden_scVI_0_1_de[["scale1", "scale2"]].max(axis=1) > 1e-4
]
leiden_scVI_0_1_de.head(20)

# %%
# Extract top 20 DE genes
# This cell extracts list of top 5 upregulated genes for every cell-type
marker_genes = (
    leiden_scVI_0_1_de.reset_index()
    .loc[lambda x: x.comparison.isin(leiden_scVI_0_1.associated_test.values)]
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
marker_genes
# %%
suffix_to_save = f"_{best_hyperparameters['dropout_rate']}_{best_hyperparameters['n_hidden']}_{best_hyperparameters['n_layers']}_{best_hyperparameters['n_latent']}_{best_hyperparameters['batch_size']}_{n_top_genes}"
marker_genes.to_csv(os.path.join(path, f"marker_genes_leiden_scVI_0.1{suffix_to_save}.csv"), index=False)

# %%
for i in range(0,len(marker_genes["comparison"].unique())):
    print("Cluster ",i)
    top_genes_cluster = marker_genes[marker_genes["comparison"] == marker_genes["comparison"].unique()[i]]["index"].tolist()#values.tolist()
    sc.pl.umap(adata, color=top_genes_cluster + ["leiden_scVI_0.1"], ncols=2, frameon=False, show=False,layer="scvi_normalized",vmin="p0.05",vmax="p99.5")


# %%