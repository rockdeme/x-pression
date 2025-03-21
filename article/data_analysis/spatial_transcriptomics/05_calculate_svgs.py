import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch


metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'filtered'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')

processed_samples = []
for s in tqdm(samples):
    adata = sc.read_h5ad(s)
    sample_name = adata.obs['sample'].values[0]
    ch.detect_svgs(adata, min_morans=0, min_spots=0.05)
    adata.write(f'{sample_folder}/{sample_name}_{sample_suffix}.h5ad')
