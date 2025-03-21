import  numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import chrysalis as ch
import decoupler as dc
import matplotlib.pyplot as plt
from data_analysis.utils import plot_adatas


# histopathological annotations for all samples
sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'filtered'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
sample_adatas = {}
for s in tqdm(samples):
    pass
    adata = sc.read_h5ad(s)
    label = adata.obs['sample'].values[0]
    sample_adatas[label] = adata
plot_adatas(sample_adatas.values(), color='annotation', alpha=1, rows=5, cols=6, size=5)
plt.show()

sample_folder = 'data/spatial_transcriptomics/h5ads'
adata = sc.read_h5ad(sample_folder + '/exp2.h5ad')

obs_df = adata.obs[['condition', 'annotation']]
freq_df = obs_df.groupby("condition")["annotation"].value_counts().unstack().fillna(0)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

freq_df_norm = freq_df.div(freq_df.sum(axis=1), axis=0)
colormaps = plt.cm.tab10.colors[: len(freq_df_norm.columns)]
fig, ax = plt.subplots(figsize=(6, 4))
bottoms = np.zeros(len(freq_df_norm))
for i, (annotation, color) in enumerate(zip(freq_df_norm.columns, colormaps)):
    ax.bar(freq_df_norm.index, freq_df_norm[annotation], bottom=bottoms, color=color, label=annotation, alpha=0.8)
    bottoms += freq_df_norm[annotation]
ax.set_ylabel("Proportion")
ax.set_title("Annotation Proportions per Condition")
ax.legend(title="Annotation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
