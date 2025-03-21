import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch
import matplotlib.pyplot as plt


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'filtered'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')

adatas = {}
for s in tqdm(samples):
    adata = sc.read_h5ad(s)
    sample_name = adata.obs['sample'][0]
    adatas[sample_name] = adata

ch.plot_svg_matrix(list(adatas.values()), figsize=(11, 8), obs_name='sample', cluster=True)
plt.show()

adata = ch.integrate_adatas(list(adatas.values()), list(adatas.keys()), sample_col='chr_sample_id')
print(adata)

adata.write(f'{sample_folder}/dataset.h5ad')
