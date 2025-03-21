import os
import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_analysis.utils import segment_tissue, plot_adatas


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
write_suffix = 'filtered'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')

processed_samples = []
for s in tqdm(samples):
        adata = sc.read_h5ad(s)
        sample_name = adata.obs['sample'].values[0]
        if not os.path.isfile(f'{sample_folder}/{sample_name}_{write_suffix}.h5ad'):
            adata = adata[adata.obs['in_tissue'] == 1]
            # adata = adata[adata.obs['annotation'] != 'unlabeled']
            adata = adata[adata.obs['annotation'] != 'blood_vessel_lumen']
            adata = adata[adata.obs['annotation'] != 'bronchiole_lumen']
            if sample_name == 'L221206':
                adata = adata[adata.obs['annotation'] != 'unlabeled']

            adata.raw = adata
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sample_metadata = metadata_df.loc[sample_name]

            # replace spatial key
            spatial_key = list(adata.uns['spatial'].keys())[0]
            if spatial_key != sample_name:
                adata.uns['spatial'][sample_name] = adata.uns['spatial'][spatial_key]
                del adata.uns['spatial'][spatial_key]

            # segment tissue
            img = adata.uns['spatial'][sample_name]['images']['hires']
            segmented_img, _, _ = segment_tissue(img, scale=1, l=20, h=30)
            adata.uns['spatial'][sample_name]['images']['hires'] = segmented_img

            sc.pp.filter_cells(adata, min_counts=sample_metadata['min_counts'])
            sc.pp.filter_cells(adata, min_genes=sample_metadata['min_genes'])

            sc.pp.normalize_total(adata, inplace=True, target_sum=1e4, exclude_highly_expressed=True)
            sc.pp.log1p(adata)

            if sample_metadata['exclude'] != 'yes':
                processed_samples.append(adata)
                adata.write(f'{sample_folder}/{sample_name}_{write_suffix}.h5ad')
        else:
            print(f'{sample_name} already exists, skipping...')

plot_adatas(processed_samples, color='total_counts')
plt.show()
plot_adatas(processed_samples, color=None, alpha=0)
plt.show()

