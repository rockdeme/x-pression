import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from data_analysis.utils import matrixplot

features_df = pd.read_csv('data/spatial_transcriptomics/image_features_random.csv')
compartment_df = features_df[[x for x in features_df.columns if 'class_' in x]]

to_remove = [x for x in features_df.columns if 'class' in x] + ['slice_idx', 'patch_idx', 'class',
                                                                'fractal_dimension', 'predicted_class']
img_df = features_df.drop(columns=to_remove)


corr_matrix = np.empty((len(img_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(img_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(img_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     # index=[cell_type_dict[x] for x in celltypes_df.columns],
                     index=img_df.columns,
                     columns=[x for x in range(compartment_df.shape[1])]).T

sns.heatmap(corrs.T, square=True)
plt.tight_layout()
plt.show()

mapping_dict = {
    'log_feature': 'Log feature',
    'skewness': 'Skewness',
    'dissimilarity': 'Dissimilarity',
    'correlation': 'Correlation',
    'edge_density': 'Edge density',
    'contrast': 'Contrast',
    'homogeneity': 'Homogeneity',
    'ASM': 'ASM',
    'energy': 'Energy',
    'std_intensity': 'STD intensity',
    'mean_intensity': 'Mean intensity',
    'wavelet_energy': 'Wavelet energy',
    'fourier_energy': 'Fourier energy',
    'circularity': 'Circularity',
    'elongation': 'Elongation'
}


corrs.columns = [mapping_dict[x] for x in corrs.columns]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

matrixplot(corrs, figsize=(5, 4), flip=True, scaling=False, square=True,
            colorbar_shrink=0.35, colorbar_aspect=8, title='Correlation of compartments\nwith micro-CT features',
            dendrogram_ratio=0.1, cbar_label="Score", ylabel='Compartments',
            cmap='magma',
            xlabel='Morphological features', rasterized=True, seed=42, reorder_obs=True,
            color_comps=True, xrot=90, ha='center')
plt.tight_layout()
plt.savefig('data/figures/morph_features_heatmap.svg')
plt.show()
