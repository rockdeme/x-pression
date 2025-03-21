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

base_path = r"C:\Users\emari\OneDrive - Universitaet Bern\Downloads_Cloud_Unibern\Demeter_Etori_Project_10X_single_cell\20230705_mouse_lung"
samples = os.listdir(base_path)
samples

# %%
# Load the files from the filtered_feature_bc_matrix.h5 from the different samples folders in a list

datasets = []
def read_rna_data(base_path, samples):
    for sample in tqdm(samples):
        print("Loading sample: ", sample)
        file = os.path.join(base_path, sample, "filtered_feature_bc_matrix.h5")

        adata = sc.read_10x_h5(file)
        adata.var_names_make_unique()
        adata.layers["raw_counts"] = adata.X.copy()

        adata.obs["sample"] = sample
        adata.obs["species"] = "mouse"
        adata.obs["tissue"] = "lung"
        # Split the 'gene_ids' column into three columns
        split_columns = adata.var["gene_ids"].str.split("_", expand=True)

        # Assign the split columns back to the dataframe
        adata.var["col_0"] = split_columns[0]
        adata.var["col_1"] = split_columns[1]
        adata.var["col_2"] = split_columns[2]
        gene_symbols_mm10 = pd.Series(adata.var_names).str.extract(r'mm10__(.*)')[0]
        gene_symbols_scov2 = pd.Series(adata.var_names).str.extract(r'scov2_gene-(.*)')[0]
        # Combine the results, prioritizing non-NaN values
        gene_symbols = gene_symbols_mm10.combine_first(gene_symbols_scov2)

        # Add the gene symbols as a new column in the dataframe
        adata.var['gene_symbols'] = gene_symbols.values
        adata.var.index = adata.var["gene_symbols"]
        adata.var['gene_symbols'] = gene_symbols.values


        datasets.append(adata)
    
    return datasets
# %%
# Load the data
datasets = read_rna_data(base_path, samples)

# %%
# Check dataset is not empty
def check_empty_datasets(datasets):
    for i,adata in tqdm(enumerate(datasets)):
        print(adata.obs["sample"].iloc[0])
        if adata.shape[0] > 0:
            print("\nSample ",adata.obs["sample"].iloc[0]," is not empty\n")
        else:
            print("\nSample ",adata.obs["sample"].iloc[0]," is empty!!!!!!!!!!!!!! ")        

# %%
check_empty_datasets(datasets)


# %%
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("Ttyh")]
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("Cd8")]
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("Fcg")]
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("Ms4a")]
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("S100a8")]
# datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith("Csf3")]


# %%
# Sanity check
for adata in datasets:
    print(f"Number of cells for sample: {adata.obs['sample'].iloc[0]} is: {adata.n_obs}")
    print(f"Number of genes for sample: {adata.obs['sample'].iloc[0]} is: {adata.n_vars}")
    print(f"Max sum of adata.X per gene for sample: {adata.obs['sample'].iloc[0]} is: {np.max(np.sum(adata.X, axis=1))}")
    print(f"Max sum of adata.X per cell for sample: {adata.obs['sample'].iloc[0]} is: {np.max(np.sum(adata.X, axis=0))}")
    print("If no decimals we are good to go")

# %%
mt_genes = datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.contains(r'^mt-', regex=True)]
#datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.startswith('mt-')]
rps_genes = datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.contains(r'^Rps', regex=True)]
hb_genes = datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.contains(r'^Hb[(a-z)]', regex=True)]
scov_genes = datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.contains(r'^GU280', regex=True)]

# %%

def qc_metrics_mt_rp_hb(adata):
    adata.var['mt'] = adata.var["gene_symbols"].isin(mt_genes.values)
    adata.var['ribo'] = adata.var["gene_symbols"].isin(rps_genes.values)
    adata.var['hb'] = adata.var["gene_symbols"].isin(hb_genes.values)
    adata.var['scov'] = adata.var["gene_symbols"].isin(scov_genes.values)

    print("Done for sample: ",str(adata.obs["sample"].iloc[0]))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb", "scov"], percent_top=[20,50,100,200,300,400,500,1000,3000,5000,], log1p=True, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4)
    sc.pl.violin(adata, ['pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb', 'pct_counts_scov'], jitter=0.4, rotation=90)
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")
    print(adata.obs)


# Run function for all datasets
for adata in datasets:
    qc_metrics_mt_rp_hb(adata)

# %%
####################################
#        Cell cycle scoring        #
####################################

def load_cell_cycle_genes(file_path):
    with open(file_path) as f:
        cell_cycle_genes = [x.strip() for x in f]
        cell_cycle_genes = [gene.capitalize() for gene in cell_cycle_genes]

    return cell_cycle_genes[:43], cell_cycle_genes[43:]

# Optional
def check_missing_genes(datasets, s_genes, g2m_genes):
    for dataset in datasets:
        missing_s_genes = [gene for gene in s_genes if gene not in dataset.var_names]
        missing_g2m_genes = [gene for gene in g2m_genes if gene not in dataset.var_names]
        print("Number of genes in s_genes: ",len(s_genes))
        print("Number of genes in g2m_genes: ",len(g2m_genes))
        print(f"For dataset: {dataset.obs['sample'].iloc[0]}, missing s_genes in adata: {missing_s_genes}")
        print(f"For dataset: {dataset.obs['sample'].iloc[0]}, missing g2m_genes in adata: {missing_g2m_genes}")

def score_cell_cycle(datasets, s_genes, g2m_genes):
    for adata in datasets:
        print(f"Cell cycle scoring for dataset: {adata.obs['sample'].iloc[0]}")
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        sc.pl.violin(adata, ['S_score', 'G2M_score'], groupby='phase', rotation=90, multi_panel=True)

# Optional, can be run better after processing.
def plot_cell_cycle_scores_reduced_adata(datasets, file_path):
    for adata in datasets:

        # reload cell cycle genes
        with open(file_path) as f:
            cell_cycle_genes = [x.strip() for x in f]

        print(f"Cell cycle scores for subset of cell cycle genes datasets: {adata.obs['sample'].iloc[0]}")
        
        valid_genes = list(set(cell_cycle_genes) & set(adata.var_names))
        if valid_genes:
            adata_cc_genes = adata[:, valid_genes].copy()
            print("Scatter")
            sc.pl.scatter(adata_cc_genes, x='S_score', y='G2M_score', color='phase')         
            adata_cc_genes.layers["raw_counts"] = adata_cc_genes.X.copy()
            # Normalize
            sc.pp.normalize_total(adata_cc_genes, target_sum=1e4)
            adata_cc_genes.layers["normalized_1e4"] = adata_cc_genes.X.copy()
            # Log
            sc.pp.log1p(adata_cc_genes)
            adata_cc_genes.layers["logcounts"] = adata_cc_genes.X.copy()
            # Scale
            print("PCA")
            sc.tl.pca(adata_cc_genes)
            sc.pl.pca_scatter(adata_cc_genes, color='phase',components=['1,2','1,3','2,3',])#'1,4','2,4','3,4'])
        else:
            print(f"No valid genes found in dataset: {adata.obs['sample'].iloc[0]}")

# %%
# Select and run functions for specific requirements, prepare paths and check list of genes.
# Use as you want:
#file_path = '/data/projects/p918_Horse_Dendritic_Cells_Comparison/Alignment_STARsolo_samples/Reference/CellCycle/regev_lab_cell_cycle_genes_FIX.txt'
file_path = r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\regev_lab_cell_cycle_genes_Fix_mouse_November_2024.txt"
# use the fix where I correct the genes that have updated SYMBOL names
# Mlf1ip -> Cenpu
s_genes, g2m_genes = load_cell_cycle_genes(file_path)
check_missing_genes(datasets, s_genes, g2m_genes)
# (I recommend ruuning this function after processing and generating the adatas cleaned, so can compare actually this too) 
score_cell_cycle(datasets, s_genes, g2m_genes)


# %%
# Add symbols for the scov genes
scov_genes = datasets[0].var["gene_symbols"][datasets[0].var["gene_symbols"].str.contains(r'^GU280', regex=True)]
scov_genes

# transform to  scov gene symbols
#https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2/
scov_symbols = {
    "GU280_gp01": "ORF1ab",
    "GU280_gp02": "S",
    "GU280_gp03": "ORF3a",
    "GU280_gp04": "E",
    "GU280_gp05": "M",
    "GU280_gp06": "ORF6",
    "GU280_gp07": "ORF7a",
    "GU280_gp08": "ORF7b",
    "GU280_gp09": "ORF8",
    "GU280_gp10": "N",
    "GU280_gp11": "ORF10",   
}

#change names of scov_genes
for adata in datasets:
    adata.var["gene_symbols_copy"] = adata.var["gene_symbols"]
    adata.var["gene_symbols"] = adata.var["gene_symbols"].replace(scov_symbols)
    adata.var_names = adata.var_names.to_series().replace(scov_symbols).values


# %%


# %%
# Raw check and reports of the data, PCA based and UMAP representation
def quality_check_raw(datasets):
    for adata in tqdm(datasets):
        adata_copy = adata.copy() #Create a copy of the data to work with
        adata_copy.X = adata_copy.layers["raw_counts"].copy()
        #Normalize
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        adata_copy.layers["normalized_1e4"] = adata_copy.X.copy()
        #Log
        sc.pp.log1p(adata_copy)
        adata_copy.layers["logcounts"] = adata_copy.X.copy()
        sc.pp.highly_variable_genes(adata_copy,)
                                    # min_mean=0.0125,
                                    # max_mean=3,
                                    # min_disp=0.5)#,subset=True)
        print("Running PCA for sample: " + str(adata_copy.obs["sample"].iloc[0]))
        sc.tl.pca(adata_copy)
        sc.pp.neighbors(adata_copy,n_pcs=30)
        print("Running UMAP for sample: " + str(adata_copy.obs["sample"].iloc[0]))
        sc.tl.umap(adata_copy)
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
                                     ],ncols=3)#'batch', 'celltypes', 'tissue', 'method', 'Species', 'Method', 'Method_version'
        print("Quality checking of the raw datasets complete.")
        del adata_copy
# %%
# Run the quality check
quality_check_raw(datasets)


# %%
# Reorder the phases
def reorder_phases(adata):
    adata.obs['phase'] = adata.obs['phase'].astype('category')
    adata.obs['phase'] = adata.obs['phase'].cat.reorder_categories(['G1', 'S', 'G2M'], ordered=True)

# Run the function for all datasets
for adata in datasets:
    reorder_phases(adata)

# Run again quality_check_raw(datasets)
# %%
quality_check_raw(datasets)

# %%
# Continue with the analysis
# Need to filter the data from low quality cells (genes,counts,mt,hb), NMADS and run again the quality check

# %%
# Check the number of cells that would be removed based on the quality control metrics

def calculate_nmad_thresholds(data, metric, nmads):
      median_value = np.median(data[metric])
      mad_value = median_abs_deviation(data[metric])
      thresholds = [(median_value - nmad * mad_value, 
                     median_value + nmad * mad_value) for nmad in nmads]
      return thresholds


# %%
#####################
#    MT filtering   #
#####################
print("Mitocondrial filtering")
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    for i in [5,10,15,20]:
        print(f"Number of cells without filtering MT > {i} is: {adata[adata.obs["pct_counts_mt"] < i].n_obs}")

# %%
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    nmads = [3, 5, 8]
    thresholds = calculate_nmad_thresholds(adata.obs, "pct_counts_mt", nmads)
    for nmad, (lower, upper) in zip(nmads, thresholds):
        kept_cells = adata[(adata.obs["pct_counts_mt"] >= lower) & (adata.obs["pct_counts_ribo"] <= upper)].n_obs
        print("Nmad: ",nmad)
        print(f"Number of cells within {nmad} NMADs is: {kept_cells}")
        print(f'Lower bound: {lower} removes {adata[adata.obs["pct_counts_mt"] <= lower].n_obs} cells')
        print(f'Upper bound: {upper} removes {adata[adata.obs["pct_counts_mt"] >= upper].n_obs} cells')



# %%
#####################
#    HB filtering   #
#####################
print("Hemoglobin filtering")
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    for i in [5,10,15,20]:
        print(f"Number of cells without filtering HB > {i} is: {adata[adata.obs["pct_counts_hb"] < i].n_obs}")
# %%
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    nmads = [3, 5, 8]
    thresholds = calculate_nmad_thresholds(adata.obs, "pct_counts_hb", nmads)
    for nmad, (lower, upper) in zip(nmads, thresholds):
        kept = adata[(adata.obs["pct_counts_hb"] >= lower) & (adata.obs["pct_counts_hb"] <= upper)].n_obs
        print("Nmad: ",nmad)
        print(f"Number of  cells within {nmad} NMADs is: {kept}")
        print(f'Lower bound: {lower} removes {adata[adata.obs["pct_counts_hb"] <= lower].n_obs} cells')
        print(f'Upper bound: {upper} removes {adata[adata.obs["pct_counts_hb"] >= upper].n_obs} cells')



# %%
#####################
#   RIBO filtering  #
#####################
print("Ribosomal filtering")
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    for i in [20,30, 40]:
        print(f"Number of cells without filtering Ribo > {i} is: {adata[adata.obs["pct_counts_ribo"] < i].n_obs}")


# %%
for adata in tqdm(datasets):
    print(f"Sample: {adata.obs['sample'].iloc[0]}")
    print(f"Number of cells without filtering is: {adata.n_obs}")
    nmads = [3, 5, 8]
    thresholds = calculate_nmad_thresholds(adata.obs, "pct_counts_ribo", nmads)
    for nmad, (lower, upper) in zip(nmads, thresholds):
        kept_cells = adata[(adata.obs["pct_counts_ribo"] >= lower) & (adata.obs["pct_counts_ribo"] <= upper)].n_obs
        print("Nmad: ",nmad)
        print(f"Number of cells within {nmad} NMADs is: {kept_cells}")
        print(f'Lower bound: {lower} removes {adata[adata.obs["pct_counts_ribo"] <= lower].n_obs} cells')
        print(f'Upper bound: {upper} removes {adata[adata.obs["pct_counts_ribo"] >= upper].n_obs} cells')



# %%
# Preprocess the data based on the quality control metrics
total_cells = 0
def pp(adata):
    total_cells_in = adata.n_obs
    
    #uppeer_lim: 97th percentile of n_genes_by_counts
    # sns.histplot(adata.obs["total_counts"], bins=100, kde=False)
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .99)
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]


    # Statistically 1 nMAD is around 1.4826 times the median of the absolute deviation from the median.
    # Which corresponds to 68% of the data.
    # 2 nMADs corresponds to 95% of the data.
    # 3 nMADs corresponds to 99.7% of the data.
    # 4 nMADs corresponds to 99.99% of the data.
    # 5 nMADs corresponds to 99.9999% of the data.
    # 6 nMADs corresponds to 99.9999999% of the data.
    # 7 nMADs corresponds to 99.9999999999% of the data.
    # 8 nMADs corresponds to 99.9999999999999% of the data.
    # 9 nMADs corresponds to 99.9999999999999999% of the data.
    # 10 nMADs corresponds to 99.9999999999999999999% of the data.
    # Statistically: 1nmad:68, 2nmad:95 and 3nmad:99.7
    # def is_outlier(adata, metric: str, nmads: int):
    #     M = adata.obs[metric]
    #     outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (np.median(M) + nmads * median_abs_deviation(M) < M)
    #     return outlier
    def is_outlier(adata, metric: str, lower_nmad: int, upper_nmad: int):
        M = adata.obs[metric]
        lower_bound = np.median(M) - lower_nmad * median_abs_deviation(M)
        upper_bound = np.median(M) + upper_nmad * median_abs_deviation(M)
        outlier = (M < lower_bound) | (M > upper_bound)
        return outlier

    
     #In other words, using a higher value of nmads will result in fewer data points being considered outliers.
    
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", lower_nmad=3, upper_nmad=5)
        | is_outlier(adata, "log1p_n_genes_by_counts", lower_nmad=3, upper_nmad=5)
        | is_outlier(adata, "pct_counts_in_top_20_genes", lower_nmad=3, upper_nmad=5))
    
    print(adata)
    #10% of mitochondrial genes,
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", lower_nmad=5, upper_nmad=5) | (adata.obs["pct_counts_mt"] > 10)
    #ribosomal genes
    # it is special as shown in the plot, usually low number of ribosomes and then then are cells with very large... 
    # so try to not filter too many only leave some for doublets tool too
    adata.obs["rp_outlier"] = is_outlier(adata, "pct_counts_ribo", lower_nmad=8,upper_nmad=8) #| (adata.obs["pct_counts_ribo"] > 45) 
    #5% of hemoglobin genes
    #we dont wanna be too permissive with hb genes..
    adata.obs["hb_outlier"] = is_outlier(adata, "pct_counts_hb", lower_nmad=3,upper_nmad=3) | (adata.obs["pct_counts_hb"] > 3)


    # Adding visualization where is the cut made of MADS
    # #log1p_total_counts
    # a = adata.obs['log1p_total_counts']
    # print("Median of :",str(adata.obs["sample"][0]),' = ',np.median(a))
    # print("Median - 5*MAD of :",str(adata.obs["sample"][0]),' = ',np.median(a) - 5*median_abs_deviation(a))
    # print("Median + 5*MAD of :",str(adata.obs["sample"][0]),' = ',np.median(a) + 5*median_abs_deviation(a))
    # ax = sns.displot(a)

    # plt.axvline(np.median(a) - 5 * median_abs_deviation(a))
    # plt.axvline(np.median(a) + 5 * median_abs_deviation(a))

    # plt.show()
    def plot_feature_distribution_with_outliers(ax, adata, feature_name, mad_multiplier):
        feature_values = adata.obs[feature_name]
        median_value = np.median(feature_values)
        mad_value = median_abs_deviation(feature_values)

        lower_bound = median_value - mad_multiplier * mad_value
        upper_bound = median_value + mad_multiplier * mad_value

        # Plotting on the provided axis
        sns.histplot(feature_values, ax=ax, kde=True)
        ax.axvline(median_value, color='k', linestyle='--', label='Median')
        ax.axvline(lower_bound, color='r', linestyle='-', label=f'Median - {mad_multiplier} * MAD')
        ax.axvline(upper_bound, color='g', linestyle='-', label=f'Median + {mad_multiplier} * MAD')
        ax.set_title(f'{feature_name} with {mad_multiplier} MADs')
        ax.legend()

    # Example usage for the features you are interested in
    features_to_plot = [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        "pct_counts_in_top_20_genes",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb"
    ]

    # Plot for each MAD multiplier
    for mad_multiplier in [3, 5, 8]:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        fig.suptitle(f'Feature Distributions with {mad_multiplier} MADs', fontsize=16)
        fig.tight_layout(pad=5.0, rect=[0, 0.03, 1, 0.05])

        for i, feature_name in enumerate(features_to_plot):
            plot_feature_distribution_with_outliers(axes[i // 3, i % 3], adata, feature_name, mad_multiplier)

        plt.show()

    adata.obs["author"] = adata.obs["sample"].copy()#.str.split("_", expand=True)[0]
    adata.uns["original_cells_raw_n_cells"] = len(adata)

    #adata.obs["Author"] = adata.obs["orig.ident"].str.split(" et al.", expand=True)[0]
    print("Raw data for sample: " + str(adata.obs["sample"].iloc[0]))
    print(f"Total number of cells in ",str(adata.obs["sample"].iloc[0])," = ", {adata.n_obs})
    
    # # # Filter out doublets
    # # print("Filtering doublets for sample: " + str(adata.obs["sample"][0]))
    # adata = adata[adata.obs['doublet_info'] == 'singlet',:].copy()
    # adata_doublets = adata[adata.obs['doublet_info'] == 'doublet',:].copy()
    # # # save the doublets for each sample in a file for future reference
    # # adata_doublets.write_h5ad("C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\02_Load_and_preprocess\adatas\doublets\adata_doublets_"+adata.obs["sample"].iloc[0]+"_doublets.h5ad")
    # adata.uns["doublet_cells_removed"] = len(adata_doublets)
    # del adata_doublets

       
    # Plot the filtered cells
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Plot 1: Histogram of total_counts
    sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram of Total Counts")

    # Plot 2: Violin plot for pct_counts_mt, pct_counts_ribo, pct_counts_hb
    # This requires reshaping the data
    melted_data = pd.melt(adata.obs, value_vars=['pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb'])
    sns.violinplot(x='variable', y='value', data=melted_data, ax=axes[0, 1])#, jitter=0.4)
    axes[0, 1].set_title("Violin Plot of Percentage Counts")

    # Plot 3: Violin plot for n_genes_by_counts and total_counts
    # Similar to Plot 2, reshape data if necessary
    melted_data_2 = pd.melt(adata.obs, value_vars=['n_genes_by_counts', 'total_counts'])
    sns.violinplot(x='variable', y='value', data=melted_data_2, ax=axes[1, 0])#, jitter=0.4)
    axes[1, 0].set_title("Violin Plot of Gene Counts and Total Counts")

    # Plot 4: Scatter plot of total_counts vs n_genes_by_counts
    axes[1, 1].scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"], c=adata.obs["pct_counts_mt"])
    axes[1, 1].set_xlabel("Total Counts")
    axes[1, 1].set_ylabel("N Genes by Counts")
    axes[1, 1].set_title("Scatter Plot of Total Counts vs N Genes by Counts")

    plt.tight_layout()
    plt.show()


    # Filter out cells with outlier values
    print("Filtering mt_outlier and outlier tagged cells for sample: " + str(adata.obs["sample"].iloc[0]))
    adata = adata[~adata.obs["outlier"]].copy()
    adata = adata[~adata.obs["mt_outlier"]].copy()
    adata = adata[~adata.obs["rp_outlier"]].copy()
    adata = adata[~adata.obs["hb_outlier"]].copy()
    print("Filtered, data: outliers for sample: ",str(adata.obs["sample"][0]))
   
    # Plot the filtered cells
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Plot 1: Histogram of total_counts
    sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram of Total Counts")

    # Plot 2: Violin plot for pct_counts_mt, pct_counts_ribo, pct_counts_hb
    # This requires reshaping the data
    melted_data = pd.melt(adata.obs, value_vars=['pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb','pct_counts_scov'])
    sns.violinplot(x='variable', y='value', data=melted_data, ax=axes[0, 1])#, jitter=0.4)
    axes[0, 1].set_title("Violin Plot of Percentage Counts")

    # Plot 3: Violin plot for n_genes_by_counts and total_counts
    # Similar to Plot 2, reshape data if necessary
    melted_data_2 = pd.melt(adata.obs, value_vars=['n_genes_by_counts', 'total_counts'])
    sns.violinplot(x='variable', y='value', data=melted_data_2, ax=axes[1, 0])#, jitter=0.4)
    axes[1, 0].set_title("Violin Plot of Gene Counts and Total Counts")

    # Plot 4: Scatter plot of total_counts vs n_genes_by_counts
    axes[1, 1].scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"], c=adata.obs["pct_counts_mt"])
    axes[1, 1].set_xlabel("Total Counts")
    axes[1, 1].set_ylabel("N Genes by Counts")
    axes[1, 1].set_title("Scatter Plot of Total Counts vs N Genes by Counts")

    plt.tight_layout()
    plt.show()
    
    # Increment counter
    removed_cells = total_cells_in - adata.n_obs
    print(f"Total number of filtered cells for sample: ",str(adata.obs["sample"][0])," is: ",{removed_cells})
    adata.uns["cells_removed"] = removed_cells
    #adata.uns["doublet_cells_removed"] = adata.obs.predicted_doublets.value_counts()

    # Increment counter
    global total_cells
    total_cells += removed_cells
    
    return adata

# %%
# Run function for all datasets
# Run function for every 15 samples
adatas_cleaned = []
for adata in datasets:
    adatas_cleaned.append(pp(adata))
     # Print final count
    print(f"Total number of filtered cells among all datasets is: {total_cells}")
print("Finished filtering of all datasets.")


# %%

# Run the quality check again
quality_check_raw(adatas_cleaned)


# %%
# Clean doublets and run again the quality check
# Doublets detection
def mark_doublets_scrublet(adata,filter_doublets=False):
    # Calculate cell doublet scores

    # The Scrublet algorithm is based on the idea that doublets, which are generated by cells with distinct gene expression
    # This function is used to calculate the doublet scores for each cell in the dataset.
    # It does not filter the doublets, but it marks them in the dataset.
    # This can be changed by setting the filter_doublets argument to True.

    scrub = scr.Scrublet(adata.X)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    print("Ploting the histogram that would show where are the neotypic doublets SCORE for sample ",adata.obs["sample"].iloc[0])
    #doublets, which are generated by cells with distinct gene expression 
    #(e.g., different cell types) and are expected to introduce more artifacts in downstream analyses. 
    #Scrublet can only detect neotypic doublets.

    # Plot histogram
    plt.figure(figsize=(4,3))
    plt.hist(doublet_scores, bins=50, color='black', alpha=0.5,)
    plt.axvline(x=scrub.threshold_, color='red', linestyle='dashed', linewidth=2)
    # Add threshold value as text
    plt.text(scrub.threshold_, plt.gca().get_ylim()[1]*0.9, 'Threshold: {:.2f}'.format(scrub.threshold_), color='red')
    plt.title('Doublet Score Histogram')
    plt.xlabel('Doublet Score')
    plt.ylabel('Cell Count')
    plt.show()
    print("Default function to plot the histogram of the doublets from scrublet")
    #https://github.com/swolock/scrublet/blob/master/examples/scrublet_basics.ipynb
    #If a cell’s neighbors are more similar to the simulated doublets than to the observed singlets,
    # the cell is likely to be a doublet and will have a high doublet_score.
    print("Typically shall look bimodal, with the first peak representing singlets and the second peak representing doublets.")    
    scrub.plot_histogram()
    print("Locating doublets in UMAP for sample ",adata.obs["sample"].iloc[0])
    scrub.set_embedding('UMAP', scr.get_umap(X=scrub.manifold_obs_,
                                             n_neighbors=15,
                                             min_dist=0.5))
    scrub.plot_embedding('UMAP', order_points=True)
    adata.obs['doublet_scores'] = doublet_scores
    adata.obs['predicted_doublets'] = predicted_doublets
    print("The number of predicted doublets for sample ",adata.obs["sample"].iloc[0]," is: ",adata.obs.predicted_doublets.value_counts())


    # Add in column with singlet/doublet instead of True/False
    adata.obs['doublet_info'] = adata.obs["predicted_doublets"].astype(str)
    adata.obs['doublet_info'] = adata.obs['doublet_info'].replace({'True': 'doublet', 'False': 'singlet'})

    # Print the column names
    print(adata.obs.columns)   
    #and adata
    adata
    # number of genes per cell, and which of them are doublets
    sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, groupby='doublet_info', rotation=45)
    
    # Filter out doublets
    if filter_doublets:
        print(f"\nTotal number of cells in {str(adata.obs['sample'].iloc[0])} is: {adata.n_obs}")
        adata_doublets = adata[adata.obs['doublet_info'] == 'doublet',:].copy()
        # save the doublets for each sample in a file for future reference
        adata_doublets.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\02_Load_and_preprocess\adatas\doublets\adata_doublets_"+adata.obs["sample"].iloc[0]+ ".h5ad")
        # Store the number of doublets removed
        adata.uns["doublet_cells_removed"] = len(adata_doublets)
        del adata_doublets
        
        # Filter out doublets
        adata = adata[adata.obs['doublet_info'] == 'singlet',:].copy()
        print(f"\nFiltered off, doublets for sample: {str(adata.obs['sample'].iloc[0])}")
        print(f"\nTotal number of cells in {str(adata.obs['sample'].iloc[0])} is: {adata.n_obs}")
         # Plot the filtered doublets
        print(f"\nPlot of filtered doublets for sample: {str(adata.obs['sample'].iloc[0])}")
        sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, groupby='doublet_info', rotation=45)
        adata

    else:
        print(f"\nDoublets were not filtered for sample: {str(adata.obs['sample'].iloc[0])}")

    return adata

#Run function for all datasets
######################################
# Do not filter the doublets for now #
######################################
#for adata in datasets:
for adata in adatas_cleaned:
    mark_doublets_scrublet(adata,filter_doublets=True)
    print("\nFinished marking doublets for sample: ",str(adata.obs["sample"].iloc[0]))


# %%
# Run the quality check again
# Raw check and reports of the data, PCA based and UMAP representation
def quality_check_raw(datasets):
    for adata in tqdm(datasets):
        adata_copy = adata.copy() #Create a copy of the data to work with
        adata_copy.X = adata_copy.layers["raw_counts"].copy()
        #Normalize
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        adata_copy.layers["normalized_1e4"] = adata_copy.X.copy()
        #Log
        sc.pp.log1p(adata_copy)
        adata_copy.layers["logcounts"] = adata_copy.X.copy()
        sc.pp.highly_variable_genes(adata_copy,)
                                    # min_mean=0.0125,
                                    # max_mean=3,
                                    # min_disp=0.5)#,subset=True)
        print("Running PCA for sample: " + str(adata_copy.obs["sample"].iloc[0]))
        sc.tl.pca(adata_copy)
        sc.pp.neighbors(adata_copy,n_pcs=30)
        print("Running UMAP for sample: " + str(adata_copy.obs["sample"].iloc[0]))
        sc.tl.umap(adata_copy)
        print("Plotting UMAP for sample: " + str(adata_copy.obs["sample"].iloc[0]))
        sc.pl.umap(adata_copy,color=['sample','species', 'tissue',
                                     'n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'log1p_n_genes_by_counts',
                                     "pct_counts_mt","pct_counts_hb","pct_counts_ribo","pct_counts_scov",
                                     'doublet_info',
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
                                     ],ncols=3)#'batch', 'celltypes', 'tissue', 'method', 'Species', 'Method', 'Method_version'
        print("Quality checking of the raw datasets complete.")
        del adata_copy

# %%
quality_check_raw(adatas_cleaned)


# %%
# Sanity check
for adata in datasets:
    print(f"Number of cells for sample: {adata.obs['sample'].iloc[0]} is: {adata.n_obs}")
    print(f"Number of genes for sample: {adata.obs['sample'].iloc[0]} is: {adata.n_vars}")
    print(f"Max sum of adata.X per gene for sample: {adata.obs['sample'].iloc[0]} is: {np.max(np.sum(adata.X, axis=1))}")
    print(f"Max sum of adata.X per cell for sample: {adata.obs['sample'].iloc[0]} is: {np.max(np.sum(adata.X, axis=0))}")
    print("If no decimals we are good to go")
    print("If decimals we need to check the data")
# %%
# Save the list of AnnData objects
for i,adata in enumerate(adatas_cleaned):
    adata.write_h5ad(r"C:\Users\emari\Documents\Environments\COVID_Etori_Demeter_hard_copy\CovidParty\data_analysis\02_Load_and_preprocess\adatas\filtered\adata_filtered_"+adata.obs["sample"].iloc[0]+"_filtered_mt_hb_nmads_3_8_ribo_8_singlets_marked_cellcycle.h5ad")




# %%
# Visualize the filtering results
stats_list = []

for adata in adatas_cleaned:
    sample = adata.obs["sample"].iloc[0]
    original_cells = adata.uns["original_cells_raw_n_cells"]
    current_filtered_cells = len(adata)
    cells_removed = adata.uns['cells_removed']
    doublet_cells_removed = adata.uns['doublet_cells_removed']
    
    print(sample)
    print(original_cells, current_filtered_cells, cells_removed, doublet_cells_removed)
    
    stats_list.append({
        'sample': sample,
        'original_cells_raw_n_cells': original_cells,
        'current_filtered_cells': current_filtered_cells,
        'cells_removed': cells_removed,
        'doublet_cells_removed': doublet_cells_removed
    })

df_stats_filtering = pd.DataFrame(stats_list)

# Display the DataFrame
print(df_stats_filtering)


# %%
# Plot the data

# Calculate the values needed for the stacked bar plot
#df_stats_filtering['viable_cells'] = df_stats_filtering['current_filtered_cells']

# Sort the dataframe by sample to keep consistent order in the plot
df_stats_filtering.sort_values('sample', inplace=True)

# Plot the data
fig, ax = plt.subplots(figsize=(15, 10))

# Colors
colors = {
    'Doublets Removed': '#FF6F61',  # Red
    'Cells Removed': '#6B5B95',     # Purple
    'Viable Cells': '#88B04B',      # Green
    'Original Cells': '#F7CAC9'     # Pink (used to set the overall bar length, won't be visible)
}

# Plot stacked bar
bar_width = 0.8
p1 = ax.bar(df_stats_filtering['sample'], df_stats_filtering['doublet_cells_removed'], label='Doublets Removed', color=colors['Doublets Removed'])
p2 = ax.bar(df_stats_filtering['sample'], df_stats_filtering['cells_removed'], bottom=df_stats_filtering['doublet_cells_removed'], label='Cells Removed', color=colors['Cells Removed'])
p3 = ax.bar(df_stats_filtering['sample'], df_stats_filtering['current_filtered_cells'], bottom=df_stats_filtering['doublet_cells_removed'] + df_stats_filtering['cells_removed'], label='Viable Cells', color=colors['Viable Cells'])

# Draw invisible bars to set the total length of original cells
ax.bar(df_stats_filtering['sample'], df_stats_filtering['original_cells_raw_n_cells'], color=colors['Original Cells'], alpha=0)


# # Add stars above specific bars
# for i, sample in enumerate(df_stats_filtering['sample']):
#     if sample in samples_25_bp_R1:
#         height = df_stats_filtering.loc[df_stats_filtering['sample'] == sample, 'original_cells_raw_n_cells'].values[0] + df_stats_filtering.loc[df_stats_filtering['sample'] == sample, 'cells_removed'].values[0] + df_stats_filtering.loc[df_stats_filtering['sample'] == sample, 'doublet_cells_removed'].values[0]
#         #ax.text(i, height, '*', ha='center', va='bottom', fontsize=12, color='black')
#         ax.text(i, height + 0.05 * df_stats_filtering['original_cells_raw_n_cells'].max(),
#                 '*', ha='center', va='bottom', fontsize=12, color='black')



# Adding labels and title
ax.set_xlabel('Sample')
ax.set_ylabel('Number of Cells')
#ax.set_title('Filtering Results per Sample')


# # Adding legend
handles, labels = ax.get_legend_handles_labels()
# # Add a star marker to the legend
# star_marker = plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=12, label='R1>25bp')
# handles.append(star_marker)
# labels.append('R1>25bp')
ax.legend(handles=handles, labels=labels)

# # Rotate x-axis labels for better readability and color specific labels
# for label in ax.get_xticklabels():
#     if label.get_text() in samples_25_bp_R1:
#         label.set_color('red')
plt.xticks(rotation=90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
print("Done\n"*10)
# %%
