import pandas as pd
import scanpy as sc
from tqdm import tqdm
from glob import glob
import numpy as np
import chrysalis as ch
import matplotlib.pyplot as plt
import archetypes as arch
from data_analysis.utils import ensembl_id_to_gene_symbol
from sklearn.decomposition import PCA
import seaborn as sns
from data_analysis.utils import plot_adatas

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'transferred'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
samples.sort(key=len)

sample_dict = {'_'.join(x.split('/')[-1].split('_')[:-1]): x for x in samples}

#%%
# EXP 1
# read adata
sample_path = 'data/spatial_transcriptomics/cell_type_deconvolution/exp1/cell2location_map/sp_1.h5ad'
adata = sc.read_h5ad(sample_path)

for sample in tqdm(np.unique(adata.obs['sample'])):
    pass
    ref_ad = adata[adata.obs['sample'] == sample]
    ad = sc.read_h5ad(sample_dict[sample])
    ad.obsm['cell2loc']  = pd.DataFrame(ref_ad.obsm['q05_cell_abundance_w_sf'].values,
                                        columns=ref_ad.uns['mod']['factor_names'],
                                        index=ad.obs.index)
    ad.write_h5ad(sample_dict[sample])

#%%

sample_path_1 = 'data/spatial_transcriptomics/cell_type_deconvolution/exp2/cell2location_map/sp_2_1.h5ad'
sample_path_2 = 'data/spatial_transcriptomics/cell_type_deconvolution/exp2/cell2location_map/sp_2_2.h5ad'

ad1 = sc.read_h5ad(sample_path_1)

for sample in tqdm(np.unique(ad1.obs['sample'])):
    pass
    ref_ad = ad1[ad1.obs['sample'] == sample]
    ad = sc.read_h5ad(sample_dict[sample])
    ad.obsm['cell2loc']  = pd.DataFrame(ref_ad.obsm['q05_cell_abundance_w_sf'].values,
                                        columns=ref_ad.uns['mod']['factor_names'],
                                        index=ad.obs.index)
    ad.write_h5ad(sample_dict[sample])

ad2 = sc.read_h5ad(sample_path_2)

for sample in tqdm(np.unique(ad2.obs['sample'])):
    pass
    ref_ad = ad2[ad2.obs['sample'] == sample]
    ad = sc.read_h5ad(sample_dict[sample])
    ad.obsm['cell2loc']  = pd.DataFrame(ref_ad.obsm['q05_cell_abundance_w_sf'].values,
                                        columns=ref_ad.uns['mod']['factor_names'],
                                        index=ad.obs.index)
    ad.write_h5ad(sample_dict[sample])
