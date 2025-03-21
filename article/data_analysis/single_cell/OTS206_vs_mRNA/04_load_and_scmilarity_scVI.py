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
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#To work in scimilarity 

# Load the model and adata
#adata_path = "/data/projects/p947_Parseq_tissues_Gilles_Foucras_Bos_taurus/GIT/Parseq_Bos_taurus_2024_September_Analysis/Load_and_integrate/SCVI/LP/adatas/covariates/adatas_covariates_replicate_and_condition_PBMCs_all/scvi_model__covariates_replicate_and_condition_all_no_doublets.h5ad"
adata_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\03_Load_and_integrate\adatas\SCVI\adatas_replicate_0.1_128_2_50_64_all\scvi_model_replicate_all_no_doublets.h5ad"
#model_path = "/data/projects/p947_Parseq_tissues_Gilles_Foucras_Bos_taurus/GIT/Parseq_Bos_taurus_2024_September_Analysis/Load_and_integrate/SCVI/LP/adatas/covariates/adatas_covariates_replicate_and_condition_PBMCs_all/scvi_model__covariates_replicate_and_condition_all_no_doublets/"
#model_path = r"C:\Users\emari\Documents\Environments\scimilarity_2024_local\WorkingEnvironment\adatas\scvi_model__covariates_replicate_all_no_doublets"
adata = sc.read_h5ad(adata_path)
#model = scvi.model.SCVI.load(model_path, adata,weights_only=True)
#model = scvi.model.SCVI.load(model_path, adata,accelerator="gpu")

# %%
#model

# %%
adata

# %%
latent_embedding = adata.obsm["X_scVI_latent"]

reducer = umap.UMAP(random_state=2207)
umap_embedding = reducer.fit_transform(latent_embedding)
adata.obsm["X_umap_X_scVI_latent_umap"] = umap_embedding

# This is how to use only the first two dimensions of the latent space
# #umap_embedding = reducer.fit_transform(np.column_stack((adata.obsm['X_scVI_latent'][:,0], adata.obsm['X_scVI_latent'][:,1])))
# can check the first 10 dimensions of the latent space
# #umap_embedding = reducer.fit_transform(adata.obsm['X_scVI_latent'][:,0:10])

#umap_embedding = adata.obsm['X_umap']
umap_df = pd.DataFrame({
    'UMAP1': umap_embedding[:, 0],
    'UMAP2': umap_embedding[:, 1],
    'replicate': adata.obs['replicate'],
    'sample': adata.obs['sample'],
    #'condition': adata.obs['condition'],
    'phase': adata.obs['phase'],
    'doublet_info': adata.obs['doublet_info']
})

umap_df_reshuffled = umap_df.sample(frac=1, random_state=2207).reset_index(drop=True)

# Plot UMAP
plt.figure(figsize=(10, 6))
sns.scatterplot(x='UMAP1', y='UMAP2', hue="replicate", style='sample',alpha=0.6, data=umap_df_reshuffled)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.title('umap UMAP of LP Harmony')
plt.show()
# # %%
# ##########################################################
# #           TRY NMF  and visualize the results           #
# ##########################################################
# from sklearn.decomposition import NMF
# import matplotlib.pyplot as plt

# # Determine n_components for NMF

# # Calculate reconstruction errors for different numbers of components
# errors = []
# components_range = range(2, 30)
# for n in tqdm(components_range):
#     nmf = NMF(n_components=n, init='random', random_state=42)
#     nmf.fit(latent_embedding)
#     errors.append(nmf.reconstruction_err_)

# # Plot Reconstruction Error
# plt.figure(figsize=(8, 6))
# plt.plot(components_range, errors, marker='o')
# plt.title("Reconstruction Error vs. Number of Components (NMF)")
# plt.xlabel("Number of Components")
# plt.ylabel("Reconstruction Error")
# plt.grid()
# plt.show()

# # Choose: The elbow point where the error decreases sharply but stabilizes afterward.

# # %%
# #########################
# #   RUN NMF             #
# #########################
# #nmf = NMF(n_components=5, init='random', random_state=42)
# latent_embedding = adata.obsm["X_scVI_latent"]
# nmf = NMF(n_components=10, init='random', random_state=42)
# #nmf_embedding = nmf.fit_transform(adata.X)
# nmf_embedding = nmf.fit_transform(latent_embedding)

# nmf_df = pd.DataFrame({
#     "NMF1": nmf_embedding[:, 0],
#     "NMF2": nmf_embedding[:, 1],
#     # "species": adata.obs["species"],
#     # "labels2": adata.obs["labels2"]
#     'replicates': adata.obs['replicate'],
#     'sample': adata.obs['sample'],
#     'phase': adata.obs['phase'],
# })

# # plot
# nmf_df_shuffled = nmf_df.sample(frac=1, random_state=2207).reset_index(drop=True)

# fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# sns.scatterplot(data=nmf_df_shuffled, x="NMF1", y="NMF2", hue="replicates", palette="tab10", s=10, ax=axes[0])
# axes[0].set_title("NMF of Data by Species")
# axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, markerscale=2)

# sns.scatterplot(data=nmf_df_shuffled, x="NMF1", y="NMF2", hue="phase", palette="tab10", s=10, ax=axes[1])
# axes[1].set_title("NMF of Data by Labels")
# axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, markerscale=2)

# plt.tight_layout()
# plt.show()


# %%
#adata.layers["counts"] = adata.layers["raw_counts"].copy()
# %%
sc.set_figure_params(dpi=300)
#plt.rcParams["figure.figsize"] = [6, 4]

import warnings

warnings.filterwarnings("ignore")

from scimilarity.utils import lognorm_counts, align_dataset
from scimilarity import CellAnnotation
# %%
# Download and extract the model
# model_url = "https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1"
# ! wget model_url
# model_path = "/models/model_v1.1.tar.gz"
# extracted_path = "/models/"
#model_path = "models/model_v1.1"
model_path = r"C:\Users\emari\Documents\Environments\scimilarity_2024_local\WorkingEnvironment\models\annotation_model_v1"
ca = CellAnnotation(model_path=model_path,use_gpu=True)

# %%
print("Model loaded")

# %%
# Set the adata.var_names to be Upper case
print("MOUSE: Setting var_names to upper case\n"*20)
adata.var_names = adata.var_names.str.upper()


# %%
adata = align_dataset(adata, ca.gene_order)

# %%
adata.layers["counts"] = adata.layers["raw_counts"].copy()
adata = lognorm_counts(adata)

# %%
adata.obsm["X_scimilarity"] = ca.get_embeddings(adata.X)



# %%
sc.pp.neighbors(adata, use_rep="X_scimilarity")
sc.tl.umap(adata,min_dist=0.1)
sc.pl.umap(adata, color="replicate", legend_fontsize=5)


# %%
# Alternative to scanpy
latent_embedding = adata.obsm["X_scimilarity"] # This is the latent space of the scimilarity model
reducer = umap.UMAP(random_state=2207)
umap_embedding = reducer.fit_transform(latent_embedding)
adata.obsm["X_umap_X_scimilarity"] = umap_embedding

# This is how to use only the first two dimensions of the latent space
# #umap_embedding = reducer.fit_transform(np.column_stack((adata.obsm['X_scVI_latent'][:,0], adata.obsm['X_scVI_latent'][:,1])))
# can check the first 10 dimensions of the latent space
# #umap_embedding = reducer.fit_transform(adata.obsm['X_scVI_latent'][:,0:10])

#umap_embedding = adata.obsm['X_umap']
umap_df = pd.DataFrame({
    'UMAP1': umap_embedding[:, 0],
    'UMAP2': umap_embedding[:, 1],
    'replicate': adata.obs['replicate'],
    #'condition': adata.obs['condition'],
    'phase': adata.obs['phase'],
    # 'bc1_wind': adata.obs['bc1_wind'],
    # 'bc2_wind': adata.obs['bc2_wind'],
    # 'bc3_wind': adata.obs['bc3_wind'],
    # 'bc1_well': adata.obs['bc1_well'],
    # 'bc2_well': adata.obs['bc2_well'],
    # 'bc3_well': adata.obs['bc3_well'],
    'sample': adata.obs['sample'],
    'doublet_info': adata.obs['doublet_info']
})
umap_df_reshuffled = umap_df.sample(frac=1, random_state=2207).reset_index(drop=True)
# Plot UMAP
fig, axs = plt.subplots(4, 3, figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

sns.scatterplot(x='UMAP1', y='UMAP2', hue="replicate", alpha=0.6, data=umap_df_reshuffled, ax=axs[0,0])
axs[0,0].set_title("replicate")
axs[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

sns.scatterplot(x='UMAP1', y='UMAP2', hue='phase', alpha=0.6, data=umap_df_reshuffled, ax=axs[0,1])
axs[0,1].set_title("phase")
axs[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

sns.scatterplot(x='UMAP1', y='UMAP2', hue='doublet_info', alpha=0.6, data=umap_df_reshuffled, ax=axs[0,2])
axs[0,2].set_title("doublet_info")
axs[0,2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

for i, sample in enumerate(["E_1", "E_2", "E_3", "E_4", "E_5", "E_6"]):
    row = (i // 3) + 1
    col = i % 3
    sns.scatterplot(x='UMAP1', y='UMAP2', color='grey', alpha=0.3, data=umap_df_reshuffled, ax=axs[row, col], s=10)
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='sample', alpha=0.6, data=umap_df_reshuffled[umap_df_reshuffled["sample"] == sample], ax=axs[row, col], s=10)
    axs[row, col].set_title(sample)
    axs[row, col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

sns.scatterplot(x='UMAP1', y='UMAP2', hue='sample', alpha=0.6, data=umap_df_reshuffled, ax=axs[3,0], s=10)
axs[3,0].set_title("All samples")
axs[3,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

sns.scatterplot(x='UMAP1', y='UMAP2', hue='sample', alpha=0.6, data=umap_df_reshuffled, ax=axs[3,1], s=10)
axs[3,1].set_title("All samples")
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.title('umap UMAP of SCVI')
plt.tight_layout()
plt.show()

# %%
sc.pl.umap(adata,color=["sample","replicate","phase","doublet_info"],legend_fontsize=5,ncols=3)


# %%
### Unconstrained annotation
# IT NEES THE labelled_kNN.bin FILE IN THE ANNOTATION FOLDER... 
# AND THE reference_labels.tsv FILE TOO
predictions, nn_idxs, nn_dists, nn_stats = ca.get_predictions_knn(adata.obsm["X_scimilarity"])
# df_scimilarity = pd.DataFrame({
#     "predictions": predictions.values,
#     'nn_idxs': nn_idxs[:, 0],
#     'nn_dists': nn_dists[:, 0],
#     'max_dist_from_nn_stats': nn_stats.iloc[:, 3],
# })


# %%
adata.obs["predictions_unconstrained"] = predictions.values
# %%
celltype_counts = adata.obs.predictions_unconstrained.value_counts()
celltype_counts

# %%
adata.obs.predictions_unconstrained.value_counts().head(50)

# %%
#############################################
# HERE IS THE FILTERING OF THE CELL TYPES   #
#############################################
#predictions.value_counts() and select minimum to see
filter_cells = 10
well_represented_celltypes = celltype_counts[celltype_counts > filter_cells].index

with plt.rc_context():
    fig, ax = plt.subplots(2,1 , figsize=(9, 10))
    sc.pl.umap(
        adata[adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
        color="predictions_unconstrained",
        legend_fontsize=5,
        show=False,
        ax=ax[0])
    sc.pl.umap(
        adata[adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
        color="predictions_unconstrained",
        legend_fontsize=5,
        legend_loc="on data",
        show=False,
        ax=ax[1])
    
    #plt.savefig(f"umap_adata{filter_cells}.png")
    plt.show()





# %%
# Not well represented cell types
sc.pl.umap(
    adata[~adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
    color="predictions_unconstrained",
    legend_fontsize=5,
    #legend_loc="on data"
    )


# %%
with plt.rc_context():
    plt.figure(figsize=(12, 5))
    sc.pl.umap(
        adata[adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
        color="predictions_unconstrained",
        frameon=False,
        legend_fontsize=5,
        legend_loc = "on data",
        show=False,)
    #plt.savefig(f"umap_adata{filter_cells}.png")
    plt.show()

# %%
idx2label_items = ca.idx2label.items()
descriptions = [description for _, description in idx2label_items]
unique_descriptions = set(descriptions)
unique_descriptions_list = list(unique_descriptions)

unique_descriptions_list

# ['erythroid progenitor cell',
#  'type I pneumocyte',
#  'myeloid cell',
#  'cardiac muscle cell',
#  'naive T cell',
#  'type B pancreatic cell',
#  'melanocyte',
#  'neuron',
#  'naive thymus-derived CD8-positive, alpha-beta T cell',
#  'myofibroblast cell',
#  'lung endothelial cell',
#  'fat cell',
#  'retina horizontal cell',
#  'group 3 innate lymphoid cell',
#  'central memory CD8-positive, alpha-beta T cell',
#  'IgM plasma cell',
#  'kidney distal convoluted tubule epithelial cell',
#  'IgA plasma cell',
#  'CD16-negative, CD56-bright natural killer cell, human',
#  'T follicular helper cell',
#  'Langerhans cell',
#  'parietal epithelial cell',
#  'type II pneumocyte',
#  'enteric smooth muscle cell',
#  'ON-bipolar cell',
#  'fibroblast',
#  'hematopoietic stem cell',
#  'classical monocyte',
#  'CD141-positive myeloid dendritic cell',
#  'kidney proximal convoluted tubule epithelial cell',
#  'effector memory CD4-positive, alpha-beta T cell',
#  'mucus secreting cell',
#  'inflammatory macrophage',
#  'mesodermal cell',
#  'mesothelial cell',
#  'intermediate monocyte',
#  'CD1c-positive myeloid dendritic cell',
#  'epithelial cell',
#  'mature NK T cell',
#  'squamous epithelial cell',
#  'class switched memory B cell',
#  'native cell',
#  'double negative thymocyte',
#  'CD4-positive, CD25-positive, alpha-beta regulatory T cell',
#  'double-positive, alpha-beta thymocyte',
#  'mucosal invariant T cell',
#  'effector memory CD8-positive, alpha-beta T cell, terminally differentiated',
#  'endothelial cell of artery',
#  'keratinocyte',
#  'Mueller cell',
#  'IgG plasma cell',
#  'pancreatic A cell',
#  'granulocyte',
#  'periportal region hepatocyte',
#  'CD8-positive, alpha-beta cytotoxic T cell',
#  'T-helper 17 cell',
#  'neural cell',
#  'monocyte',
#  'mesenchymal stem cell',
#  'common lymphoid progenitor',
#  'respiratory basal cell',
#  'natural killer cell',
#  'secretory cell',
#  'animal cell',
#  'B cell',
#  'CD4-positive, alpha-beta memory T cell',
#  'oligodendrocyte',
#  'CD4-positive, alpha-beta T cell',
#  'macrophage',
#  'non-classical monocyte',
#  'CD4-positive helper T cell',
#  'gamma-delta T cell',
#  'CD14-positive monocyte',
#  'blood vessel endothelial cell',
#  'bronchial smooth muscle cell',
#  'paneth cell of epithelium of small intestine',
#  'naive thymus-derived CD4-positive, alpha-beta T cell',
#  'stromal cell',
#  'platelet',
#  'lung macrophage',
#  'ionocyte',
#  'follicular B cell',
#  'enteroendocrine cell',
#  'paneth cell',
#  'germinal center B cell',
#  'plasma cell',
#  'slow muscle cell',
#  'lymphocyte',
#  'precursor B cell',
#  'innate lymphoid cell',
#  'plasmacytoid dendritic cell, human',
#  'CD4-positive, alpha-beta cytotoxic T cell',
#  'effector CD8-positive, alpha-beta T cell',
#  'smooth muscle cell of prostate',
#  'luminal cell of prostate epithelium',
#  'fibroblast of lung',
#  'glandular epithelial cell',
#  'pancreatic acinar cell',
#  'pancreatic stellate cell',
#  'alveolar macrophage',
#  'endothelial cell of lymphatic vessel',
#  'memory T cell',
#  'CD8-positive, alpha-beta memory T cell',
#  'dendritic cell',
#  'pancreatic ductal cell',
#  'mast cell',
#  'glutamatergic neuron',
#  'kidney interstitial fibroblast',
#  'immature B cell',
#  'alpha-beta T cell',
#  'tracheal goblet cell',
#  'radial glial cell',
#  'stromal cell of ovary',
#  'retinal bipolar neuron',
#  'activated CD4-positive, alpha-beta T cell',
#  'CD14-positive, CD16-positive monocyte',
#  'megakaryocyte',
#  'leukocyte',
#  'CD16-positive, CD56-dim natural killer cell, human',
#  'colon epithelial cell',
#  'cardiac neuron',
#  'regulatory T cell',
#  'follicular dendritic cell',
#  'retinal cone cell',
#  'kidney connecting tubule epithelial cell',
#  'basal cell of prostate epithelium',
#  'hematopoietic precursor cell',
#  'neuroendocrine cell',
#  'neural crest cell',
#  'luminal epithelial cell of mammary gland',
#  'microglial cell',
#  'erythrocyte',
#  'capillary endothelial cell',
#  'naive B cell',
#  'ciliated cell',
#  'interstitial cell of Cajal',
#  'hepatic stellate cell',
#  'intestinal tuft cell',
#  'duct epithelial cell',
#  'kidney loop of Henle thin descending limb epithelial cell',
#  'oligodendrocyte precursor cell',
#  'enterocyte',
#  'myeloid dendritic cell',
#  'plasmablast',
#  'club cell',
#  'CD8-positive, alpha-beta T cell',
#  'effector CD4-positive, alpha-beta T cell',
#  'cholangiocyte',
#  'goblet cell',
#  'memory B cell',
#  'retinal pigment epithelial cell',
#  'pro-B cell',
#  'mesenchymal cell',
#  'erythroid lineage cell',
#  'thymocyte',
#  'pulmonary ionocyte',
#  'progenitor cell',
#  'amacrine cell',
#  'kidney epithelial cell',
#  'glial cell',
#  'vein endothelial cell',
#  'OFF-bipolar cell',
#  'pancreatic D cell',
#  'dendritic cell, human',
#  'T-helper 1 cell',
#  'kidney collecting duct principal cell',
#  'vascular associated smooth muscle cell',
#  'T cell',
#  'fast muscle cell',
#  'medullary thymic epithelial cell',
#  'activated CD8-positive, alpha-beta T cell',
#  'retinal ganglion cell',
#  'effector memory CD8-positive, alpha-beta T cell',
#  'lung ciliated cell',
#  'kidney collecting duct intercalated cell',
#  'stem cell',
#  'epicardial adipocyte',
#  'neutrophil',
#  'endothelial cell',
#  'fibroblast of cardiac tissue',
#  'retinal rod cell',
#  'endothelial cell of hepatic sinusoid',
#  'endothelial cell of vascular tree',
#  'myeloid dendritic cell, human',
#  'astrocyte',
#  'plasmacytoid dendritic cell',
#  'smooth muscle cell',
#  'cardiac endothelial cell',
#  'pericyte',
#  'epithelial cell of proximal tubule',
#  'central memory CD4-positive, alpha-beta T cell',
#  'intestinal crypt stem cell',
#  'Kupffer cell',
#  'lung neuroendocrine cell',
#  'transit amplifying cell of small intestine',
#  'CD14-low, CD16-positive monocyte',
#  'respiratory epithelial cell',
#  'naive regulatory T cell',
#  'conventional dendritic cell',
#  'hepatocyte',
#  'lung secretory cell',
#  'basal cell',
#  'kidney loop of Henle thick ascending limb epithelial cell']
# %%
### Constrained classification
target_celltypes = [

    'progenitor cell',
    'activated CD4-positive, alpha-beta T cell',
    'fibroblast of lung',
    #'fibroblast of cardiac tissue',
    #'native cell',
    #'naive thymus-derived CD8-positive, alpha-beta T cell',
    'erythroid progenitor cell',
    'paneth cell',
    'endothelial cell',
    #'conventional dendritic cell',
    #'CD16-positive, CD56-dim natural killer cell, human',
    #'germinal center B cell',
    #'type B pancreatic cell',
    #'interstitial cell of Cajal',
    #'CD14-positive, CD16-positive monocyte',
    #'retinal rod cell',
    #'effector memory CD4-positive, alpha-beta T cell',
    'keratinocyte',
    'fat cell',
    'CD141-positive myeloid dendritic cell',
    #'basal cell of prostate epithelium',
    'T-helper 17 cell',
    'lung secretory cell',
    'erythrocyte',
    #'retinal ganglion cell',
    #'microglial cell',
    #'T follicular helper cell',
    #'stromal cell of ovary',
    #'kidney collecting duct intercalated cell',
    #'oligodendrocyte precursor cell',
    #'pancreatic stellate cell',
    #'secretory cell',
    #'cholangiocyte',
    'inflammatory macrophage',
    'B cell',
    #'slow muscle cell',
    'lung macrophage',
    #'dendritic cell, human',
    #'pancreatic ductal cell',
    #'mesothelial cell',
    #'epicardial adipocyte',
    'gamma-delta T cell',
    #'CD8-positive, alpha-beta cytotoxic T cell',
    'naive T cell',
    #'ionocyte',
    #'CD4-positive, alpha-beta cytotoxic T cell',
    #'bronchial smooth muscle cell',
    #'stem cell',
    #'cardiac neuron',
    'IgA plasma cell',
    'memory B cell',
    'pro-B cell',
    #'glutamatergic neuron',
    #'neural cell',
    #'kidney collecting duct principal cell',
    #'blood vessel endothelial cell',
    #'duct epithelial cell',
    #'neural crest cell',
    'type I pneumocyte',
    'lung endothelial cell',
    #'smooth muscle cell of prostate',
    #'Mueller cell',
    'erythroid lineage cell',
    #'epithelial cell of proximal tubule',
    'natural killer cell',
    'CD1c-positive myeloid dendritic cell',
    #'radial glial cell',
    'IgG plasma cell',
    'lung ciliated cell',
    'stromal cell',
    #'CD16-negative, CD56-bright natural killer cell, human',
    #'kidney distal convoluted tubule epithelial cell',
    'fibroblast',
    #'innate lymphoid cell',
    #'glandular epithelial cell',
    #'squamous epithelial cell',
    #'central memory CD4-positive, alpha-beta T cell',
    'naive B cell',
    'neutrophil',
   # 'pancreatic D cell',
   # 'neuron',
    #'IgM plasma cell',
   # 'cardiac muscle cell',
    'endothelial cell of lymphatic vessel',
   # 'kidney epithelial cell',
   # 'myeloid dendritic cell, human',
   # 'hepatocyte',
   # 'mesenchymal stem cell',
    'immature B cell',
    'megakaryocyte',
    'CD4-positive helper T cell',
    'vascular associated smooth muscle cell',
    'group 3 innate lymphoid cell',
    'lung neuroendocrine cell',
   # 'luminal cell of prostate epithelium',
   # 'kidney connecting tubule epithelial cell',
   # 'alpha-beta T cell',
    'hematopoietic stem cell',
    'non-classical monocyte',
    'plasmacytoid dendritic cell',
    'CD14-low, CD16-positive monocyte',
   # 'monocyte',
    #'enterocyte',
    #'retinal pigment epithelial cell',
    'hematopoietic precursor cell',
    'CD4-positive, alpha-beta T cell',
    #'central memory CD8-positive, alpha-beta T cell',
    #'oligodendrocyte',
    'CD14-positive monocyte',
    'naive regulatory T cell',
    #'basal cell',
    #'cardiac endothelial cell',
    'mature NK T cell',
    #'intestinal crypt stem cell',
    #'glial cell',
    #'OFF-bipolar cell',
    #'ON-bipolar cell',
    #'amacrine cell',
    #'parietal epithelial cell',
    #'CD4-positive, CD25-positive, alpha-beta regulatory T cell',
    'common lymphoid progenitor',
    'plasmablast',
    #'follicular B cell',
    #'mucosal invariant T cell',
    'precursor B cell',
    #'fast muscle cell',
    #'endothelial cell of hepatic sinusoid',
    'respiratory epithelial cell',
    'CD8-positive, alpha-beta T cell',
   # 'pancreatic acinar cell',
    'pulmonary ionocyte',
    'plasma cell',
   # 'astrocyte',
   # 'ciliated cell',
    'alveolar macrophage',
   # 'club cell',
   # 'retina horizontal cell',
   # 'naive thymus-derived CD4-positive, alpha-beta T cell',
    'CD4-positive, alpha-beta memory T cell',
   # 'paneth cell of epithelium of small intestine',
   # 'endothelial cell of vascular tree',
   # 'leukocyte',
   # 'kidney proximal convoluted tubule epithelial cell',
   # 'follicular dendritic cell',
   # 'pancreatic A cell',
    'respiratory basal cell',
    'effector CD4-positive, alpha-beta T cell',
    #'T cell',
    #'effector memory CD8-positive, alpha-beta T cell, terminally differentiated',
    'capillary endothelial cell',
    #'medullary thymic epithelial cell',
    #'plasmacytoid dendritic cell, human',
    'activated CD8-positive, alpha-beta T cell',
    #'myeloid dendritic cell',
    'epithelial cell',
    #'thymocyte',
    #'kidney loop of Henle thick ascending limb epithelial cell',
    #'enteroendocrine cell',
    'type II pneumocyte',
    #'enteric smooth muscle cell',
    #'regulatory T cell',
    #'retinal bipolar neuron',
    #'double-positive, alpha-beta thymocyte',
    #'endothelial cell of artery',
    #'effector memory CD8-positive, alpha-beta T cell',
    'platelet',
    #'lymphocyte',
    #'melanocyte',
    #'class switched memory B cell',
    #'mesenchymal cell',
    'macrophage',
    #'mesodermal cell',
    'mast cell',
    #'myeloid cell',
    'goblet cell',
    #'vein endothelial cell',
    #'neuroendocrine cell',
    #'tracheal goblet cell',
    #'hepatic stellate cell',
    'effector CD8-positive, alpha-beta T cell',
    #'luminal epithelial cell of mammary gland',
    'granulocyte',
    'mucus secreting cell',
    #'transit amplifying cell of small intestine',
    'CD8-positive, alpha-beta memory T cell',
    'classical monocyte',
    #'T-helper 1 cell',
    'smooth muscle cell',
    #'intestinal tuft cell',
    #'kidney loop of Henle thin descending limb epithelial cell',
    #'dendritic cell',
    #'kidney interstitial fibroblast',
    #'Kupffer cell',
    #'animal cell',
    #'colon epithelial cell',
    #'memory T cell',
    'pericyte',
    #'retinal cone cell',
    #'Langerhans cell',
    #'myofibroblast cell',
    #'double negative thymocyte',
    'intermediate monocyte',
   # 'periportal region hepatocyte'
   ]

ca.safelist_celltypes(target_celltypes)


# %%
adata = ca.annotate_dataset(adata)

# %%
with plt.rc_context():
    fig, ax = plt.subplots(2,1, figsize=(19, 14))
    sc.pl.umap(adata, color="celltype_hint", legend_fontsize=5,show=False,ax=ax[0],frameon=False)
    sc.pl.umap(adata, color="celltype_hint", legend_fontsize=5,legend_loc="on data",show=False,ax=ax[1],frameon=False)
    #plt.savefig("umap_adata_constrained_annotation.png")
    plt.show()


# %%
# Split by condition
with plt.rc_context():
    fig, axs = plt.subplots(3,1, figsize=(11, 24))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_1","E_2","E_3"])],ax=axs[0],
               color="celltype_hint",frameon=False,show=False,title="E mRNA")
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_4","E_5","E_6"])],ax=axs[1],
               color="celltype_hint",frameon=False,show=False,title="E OTS")
    sc.pl.umap(adata,ax=axs[2],
               color="celltype_hint",frameon=False,show=False,title="Merged")
    #plt.savefig("umap_adata.png")
    plt.show()

# %%
# Split by condition
with plt.rc_context():
    fig, axs = plt.subplots(3,1, figsize=(42, 38))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_1","E_2","E_3"])],ax=axs[0],
               color="celltype_hint",frameon=False,show=False,title="E mRNA",legend_loc="on data")
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_4","E_5","E_6"])],ax=axs[1],
               color="celltype_hint",frameon=False,show=False,title="E OTS",legend_loc="on data")
    sc.pl.umap(adata,ax=axs[2],
               color="celltype_hint",frameon=False,show=False,title="Merged",legend_loc="on data")
    #plt.savefig("umap_adata.png")
    plt.show()

# %%
# Split by each replicate individually
with plt.rc_context():
    fig, axs = plt.subplots(2,4, figsize=(22, 10))

    sc.pl.umap(adata[adata.obs["sample"] == "E_1"],ax=axs[0,0],color="replicate",
               vmin="p01.15",vmax="p98.5",show=False,title="replicate E1")
    sc.pl.umap(adata[adata.obs["sample"] == "E_2"],ax=axs[0,1],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E2")
    sc.pl.umap(adata[adata.obs["sample"] == "E_3"],ax=axs[0,2],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E3")
    sc.pl.umap(adata[adata.obs["sample"] == "E_4"],ax=axs[0,3],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E4")
    sc.pl.umap(adata[adata.obs["sample"] == "E_5"],ax=axs[1,0],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E5")
    sc.pl.umap(adata[adata.obs["sample"] == "E_6"],ax=axs[1,1],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E6")
    sc.pl.umap(adata, color="phase", ax=axs[1,2], show=False, frameon=False,title="phase")
    sc.pl.umap(adata,ax=axs[1,3],color="replicate",show=False,title="merged")

    #plt.savefig("umap_adata.png")
    plt.show()

    
# %%
with plt.rc_context():
    fig, axs = plt.subplots(2,4, figsize=(19,9))
    sc.pl.umap(adata, color="sample", groups=["E_1"], ax=axs[0,0], show=False, frameon=False)
    sc.pl.umap(adata, color="sample", groups=["E_2"], ax=axs[0,1], show=False, frameon=False)    
    sc.pl.umap(adata, color="sample", groups=["E_3"], ax=axs[0,2], show=False, frameon=False)    
    sc.pl.umap(adata, color="sample", groups=["E_4"], ax=axs[0,3], show=False, frameon=False)
    sc.pl.umap(adata, color="sample", groups=["E_5"], ax=axs[1,0], show=False, frameon=False)
    sc.pl.umap(adata, color="sample", groups=["E_6"], ax=axs[1,1], show=False, frameon=False)
    sc.pl.umap(adata, color="phase", ax=axs[1,2], show=False, frameon=False)
    sc.pl.umap(adata, color="replicate", ax=axs[1,3], show=False, frameon=False)

    plt.tight_layout()
    #plt.savefig("adata_harmony_model_replicate_PBMC_separated_samples_distribution.png")
    plt.show()
# %% Now use the predictions to plot the UMAP with the Harmony reduction.
sc.pp.neighbors(adata, use_rep="X_scVI_latent")
sc.tl.umap(adata,random_state=2207)
sc.pl.umap(adata, color="celltype_hint")

# %%
with plt.rc_context():
    fig, ax = plt.subplots(2,1, figsize=(14, 21))
    sc.pl.umap(adata, color="celltype_hint", legend_fontsize=5,show=False,ax=ax[0],frameon=False)
    sc.pl.umap(adata, color="celltype_hint", legend_fontsize=5,legend_loc="on data",show=False,ax=ax[1],frameon=False)
    plt.savefig("umap_adata_constrained_annotation.png")
    plt.show()


# %%
# Split by condition
with plt.rc_context():
    fig, axs = plt.subplots(3,1, figsize=(12, 21))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_1","E_2","E_3"])],ax=axs[0],
               color="celltype_hint",frameon=False,show=False,title="E mRNA")
    
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_4","E_5","E_6"])],ax=axs[1],
               color="celltype_hint",frameon=False,show=False,title="E OTS")
    
    sc.pl.umap(adata,ax=axs[2],
               color="celltype_hint",frameon=False,show=False,title="Merged")
    plt.savefig("umap_adata_constrained_annotation_split_mRNA_OTS_Merged.png")
    plt.show()

# %%
# Split by condition, legend on data.
with plt.rc_context():
    fig, axs = plt.subplots(1,3, figsize=(51, 18))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_1","E_2","E_3"])],ax=axs[0],
               color="celltype_hint",vmin="p0.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data",title="mRNA")
    
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_4","E_5","E_6"])],ax=axs[1],
               color="celltype_hint",vmin="p0.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data", title="OTS")
    
    sc.pl.umap(adata,ax=axs[2],
               color="celltype_hint",vmin="p0.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data",title="Merged")
    plt.savefig("umap_adata_constrained_annotation_split_mRNA_OTS_Merged_legend_on.png")
    plt.show()

# %%
# Split by each replicate individually
with plt.rc_context():
    fig, axs = plt.subplots(2,4, figsize=(22, 10))

    sc.pl.umap(adata[adata.obs["replicate"] == "E_1"],ax=axs[0,0],color="replicate",
               vmin="p01.15",vmax="p98.5",show=False,title="replicate E_1")
    sc.pl.umap(adata[adata.obs["replicate"] == "E_2"],ax=axs[0,1],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E_2")
    sc.pl.umap(adata[adata.obs["replicate"] == "E_3"],ax=axs[0,2],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E_3")
    sc.pl.umap(adata[adata.obs["replicate"] == "E_4"],ax=axs[0,3],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E_4")
    sc.pl.umap(adata[adata.obs["replicate"] == "E_5"],ax=axs[1,0],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E_5")
    sc.pl.umap(adata[adata.obs["replicate"] == "E_6"],ax=axs[1,1],color="replicate",
                vmin="p01.15",vmax="p98.5",show=False,title="replicate E_6")
    sc.pl.umap(adata,ax=axs[1,2],color="phase",
                show=False,title="phase")
    sc.pl.umap(adata,ax=axs[1,3],color="replicate",
                show=False,title="merged")
    plt.savefig("umap_adata_constrained_annotation_split_sample.png")
    plt.show()

    
# %%
with plt.rc_context():
    fig, axs = plt.subplots(2,4, figsize=(19,9))
    sc.pl.umap(adata, color="replicate", groups=["E_1"], ax=axs[0,0], show=False, frameon=False,title="E_1")
    sc.pl.umap(adata, color="replicate", groups=["E_2"], ax=axs[0,1], show=False, frameon=False,title="E_2")    
    sc.pl.umap(adata, color="replicate", groups=["E_3"], ax=axs[0,2], show=False, frameon=False,title="E_3")    
    sc.pl.umap(adata, color="replicate", groups=["E_4"], ax=axs[0,3], show=False, frameon=False,title="E_4")
    sc.pl.umap(adata, color="replicate", groups=["E_5"], ax=axs[1,0], show=False, frameon=False,title="E_5")
    sc.pl.umap(adata, color="replicate", groups=["E_6"], ax=axs[1,1], show=False, frameon=False,title="E_6")
    #sc.pl.umap(adata, color="replicate", groups=["E_3"], ax=axs[1,2], show=False, frameon=False)
    sc.pl.umap(adata, color="phase", ax=axs[1,2], show=False, frameon=False,title="phase")
    sc.pl.umap(adata, color="sample", ax=axs[1,3], show=False, frameon=False,title="sample")
    plt.tight_layout()
    plt.savefig("umap_adata_constrained_annotation_split_samples_distribution.png")
    plt.show()

# %%
#######################################
#      UNCONSTRAINED AGAIN            #
#######################################

#############################################
# HERE IS THE FILTERING OF THE CELL TYPES   #
#############################################
#predictions.value_counts() and select minimum to see
for filter_cells in [10,40,100,200]:
    well_represented_celltypes = celltype_counts[celltype_counts > filter_cells].index

    with plt.rc_context():
        fig, ax = plt.subplots(2,1 , figsize=(9, 10))
        sc.pl.umap(
            adata[adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
            color="predictions_unconstrained",
            legend_fontsize=5,
            show=False,
            ax=ax[0])
        sc.pl.umap(
            adata[adata.obs.predictions_unconstrained.isin(well_represented_celltypes)],
            color="predictions_unconstrained",
            legend_fontsize=5,
            legend_loc="on data",
            show=False,
            ax=ax[1])

        plt.savefig(f"umap_adata_unconstrained_{filter_cells}.png")
        plt.show()


# %%
#AID=AICDA
#APE1 = APEX1
#BLIMP1 = PRDM1
# CD20 = MS4A1
# Germinal center genes
print("Merged samples")
with plt.rc_context():
    #fig, ax = plt.subplots(2,1, figsize=(9, 10))
    sc.pl.umap(adata,color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],vmin="p0.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data",title="Merged")
    plt.savefig("umap_adata.png")
    plt.show()
#sc.pl.umap(adata,color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],vmin="p01.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data",title="Merged")

# %%

print("Only mRNA samples")
with plt.rc_context():
    #fig, ax = plt.subplots(2,1, figsize=(9, 10))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_1","E_2","E_3"])],
               color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],
               vmin="p01.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data")
    plt.savefig("umap_adata_mRNA_samples_only_gene_markers.png")
    plt.show()

# sc.pl.umap(adata[adata.obs["sample"].isin(["C1_PBMC","C2_PBMC","C3_PBMC"])],
#                color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],
#                vmin="p01.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data")
    
# %%
print("Only OTS samples")
with plt.rc_context():
    #fig, ax = plt.subplots(2,1, figsize=(9, 10))
    sc.pl.umap(adata[adata.obs["sample"].isin(["E_4","E_5","E_6"])],
               color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],
               vmin="p01.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data")
    plt.savefig("umap_adata_OTS_samples_only_gene_markers.png")
    plt.show()

# sc.pl.umap(adata[adata.obs["sample"].isin(["T1_PBMC","T2_PBMC","T3_PBMC","T4_PBMC"])],
#            color=["CD14","FCGR3A","FLT3","CD3D","CD4","CD8A","MKI67","CADM1","XCR1","CXCR3","BLNK","TCF4","CD163","CD34","CXCR4","CXCR5","BCL6","CD40","AICDA","APEX1","UNG","CD79A","CD79B","MS4A1","MYC","FOXO1","XBP1","PRDM1","PAX5","BCL2"],
#            vmin="p01.15",vmax="p98.5",frameon=False,show=False,legend_loc="on data")
    
# %%
adata[adata.obs.predictions_unconstrained.isin(celltype_counts[celltype_counts > 10].index)].obs['leiden_scVI_0.3'].value_counts()

# %%
for i in [5,10,40,100,200]:
    print('Number of cells using ',i, ' as treshold is: ', adata[adata.obs.predictions_unconstrained.isin(celltype_counts[celltype_counts > i].index)].n_obs)
    print("Number of cells per cluster: \n",adata[adata.obs.predictions_unconstrained.isin(celltype_counts[celltype_counts > i].index)].obs['leiden_scVI_0.3'].value_counts())

# %%
# #!pip install nbformat plotly

# import plotly.express as px

# # Assuming adata is your AnnData object
# # Create a DataFrame from the AnnData object
# df = adata.obs.copy()
# df['UMAP1'] = adata.obsm['X_umap'][:, 0]
# df['UMAP2'] = adata.obsm['X_umap'][:, 1]

# # Create an interactive plot using plotly
# fig = px.scatter(
#     df,
#     x='UMAP1',
#     y='UMAP2',
#     color='celltype_hint',
#     title='UMAP of Cell Types',
#     labels={'color': 'Cell Type'},
#     hover_data=['celltype_hint']
# )

# # Save the plot as an HTML
# fig.write_html('umap_plot.html')
# # Show the plot
# #fig.show()
# # %%
# import plotly.graph_objects as go

# # Assuming adata is your AnnData object
# # Create a DataFrame from the AnnData object
# df = adata.obs.copy()
# df['UMAP1'] = adata.obsm['X_umap'][:, 0]
# df['UMAP2'] = adata.obsm['X_umap'][:, 1]

# # Create a scatter plot using plotly.graph_objects
# fig = go.Figure()

# # Add scatter traces for each cell type
# for cell_type in df['celltype_hint'].unique():
#     cell_type_df = df[df['celltype_hint'] == cell_type]
#     fig.add_trace(go.Scatter(
#         x=cell_type_df['UMAP1'],
#         y=cell_type_df['UMAP2'],
#         mode='markers',
#         name=cell_type,
#         text=cell_type_df['celltype_hint'],
#         hoverinfo='text',
#         marker=dict(size=5),
#         showlegend=True
#     ))

# # Update layout for better interactivity
# fig.update_layout(
#     title='UMAP of Cell Types',
#     xaxis_title='UMAP1',
#     yaxis_title='UMAP2',
#     legend_title='Cell Type',
#     legend=dict(itemsizing='constant')
# )

# # Add interactivity for highlighting points when hovering over the legend
# fig.update_traces(marker=dict(opacity=0.6), selector=dict(mode='markers'))
# fig.update_layout(legend_itemclick='toggleothers', legend_itemdoubleclick='toggle')

# # Save the plot as an HTML
# fig.write_html('umap_plot_constrained_scmilarity_humans LP Harmony_hoverable.html')

# # # Show the plot
# # fig.show()

# %%
adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\E1_E6\04_Load_and_scimilarity\adata_preprocessed_integrated_scVI_annotated_scimilarity.h5ad")
print("Saved adata with scimilarity")

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

model = models.Model.load(model = "Lethal_COVID19_Lung.pkl")
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
predictions = celltypist.annotate(adata, model = 'Lethal_COVID19_Lung.pkl', majority_voting = True,use_GPU=True)
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
adata.obs.celltype_hint = adata.obs.celltype_hint.replace({"stromal cell":"Stromal cell","endothelial cell of lymphatic vessel":"Endothelial cell of lymphatic vessel","vascular associated smooth muscle cell":"Vascular associated smooth muscle cell","respiratory epithelial cell":"Respiratory epithelial cell","pulmonary ionocyte":"Pulmonary ionocyte","plasma cell":"Plasma cell","effector CD8-positive, alpha-beta T cell":"Effector CD8-positive, alpha-beta T cell","granulocyte":"Granulocyte","mucus secreting cell":"Mucus secreting cell","CD8-positive, alpha-beta memory T cell":"CD8-positive, alpha-beta memory T cell","classical monocyte":"Classical monocyte","smooth muscle cell":"Smooth muscle cell","pericyte":"Pericyte","intermediate monocyte":"Intermediate monocyte"})
