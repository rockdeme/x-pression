import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch
import matplotlib.pyplot as plt


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'

adata = sc.read_h5ad(f'{sample_folder}/dataset.h5ad')

ch.plot_explained_variance(adata)
plt.show()

ch.aa(adata, n_pcs=20, n_archetypes=10, max_iter=1)

ch.plot_samples(adata, rows=3, cols=4, dim=10, suptitle='Uncorrected', sample_col='chr_sample_id', show_title=True)


import chrysalis as ch
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

ch.detect_svgs(adata)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

ch.pca(adata)

ch.aa(adata, n_pcs=20, n_archetypes=8)

ch.plot(adata)
plt.show()