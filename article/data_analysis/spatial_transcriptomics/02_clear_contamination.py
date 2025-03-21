import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def plot_spatial_histology_and_gene_expression(source_adatas, gene, figsize=(8, 4)):
    num_datasets = len(source_adatas)
    fig, axs = plt.subplots(num_datasets, 2, figsize=(figsize[0], figsize[1] * num_datasets))

    for dataset_idx, adata in enumerate(source_adatas):
        # Plot histology (no color) on the left
        sc.pl.spatial(adata, color=None, s=16, ax=axs[dataset_idx, 0], show=False)
        axs[dataset_idx, 0].set_title(f"Dataset {dataset_idx + 1} - Histology")

        # Plot gene expression on the right for the specific gene
        sc.pl.spatial(adata, color=gene, s=16, ax=axs[dataset_idx, 1], show=False)
        axs[dataset_idx, 1].set_title(f"Dataset {dataset_idx + 1} - Gene {gene}")

    plt.tight_layout()
    plt.show()


def get_sum_counts(adatas: list):

    genes = list(adatas[0].var_names)  # we assume adatas with equal var_names
    spots = list(adatas[0].obs_names)  # we assume adatas with equal obs_names

    gene_arr = np.zeros((len(spots), len(genes)))
    for ad in tqdm(adatas):
        gene_arr += ad.X

    gene_sums_df = pd.DataFrame(gene_arr)
    gene_sums_df.index = spots
    gene_sums_df.columns = genes

    return gene_sums_df


def get_fractions(adatas: dict, selected_adata: str):
    local_adatas = adatas.copy()
    selected_sample = adatas[selected_adata].copy()
    # local_adatas.pop(selected_adata, None)  # we don't need this probably

    genes = list(selected_sample.var_names)
    spots = list(selected_sample.obs_names)

    gene_sums_df = get_sum_counts(list(local_adatas.values()))  # move this out of the function

    gene_fractions = selected_sample.X.toarray() / (gene_sums_df.values+ 1e-8)
    gene_fractions_df = pd.DataFrame(gene_fractions)
    gene_fractions_df.index = spots
    gene_fractions_df.columns = genes

    return gene_fractions_df, gene_sums_df


def calculate_avg_contamination_fraction(adatas: dict, mock_adatas: list, scov2_genes):
    import scipy.stats as stats

    gene_means = []
    for mock in mock_adatas:
        frac_df, sum_df = get_fractions(adatas, mock)
        spots_in_tissue = list(adatas[mock].obs[adatas[mock].obs['in_tissue'] == 1].index)
        # frac_df = frac_df.loc[spots_in_tissue]
        # sum_df = sum_df.loc[spots_in_tissue]
        covid_frac = frac_df[scov2_genes]
        [gene_means.append(x) for x in covid_frac.mean(axis=0).values]

    mean = np.mean(gene_means)
    std_dev = np.std(gene_means, ddof=1)
    n = len(gene_means)

    # 95CI
    # Calculate the t-critical value for 95% CI
    t_crit = t.ppf(0.975, df=n - 1)  # 0.975 for two-tailed 95% CI
    margin_of_error = t_crit * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    print(f"95% CI: {lower_bound:.4f}, {mean:.4f}, {upper_bound:.4f}")

    # contamination = np.mean(gene_means)
    return upper_bound


def calculate_contamination_fraction(adatas: dict, mock_adatas: list, scov2_genes):
    import scipy.stats as stats

    fractions = []
    counts = []
    # gene_means = []
    spot_means = []
    spot_fractions = []

    for mock in mock_adatas:
        frac_df, sum_df = get_fractions(adatas, mock)
        spots_in_tissue = list(adatas[mock].obs[adatas[mock].obs['in_tissue'] == 1].index)
        # frac_df = frac_df.loc[spots_in_tissue]
        # sum_df = sum_df.loc[spots_in_tissue]
        covid_frac = frac_df[scov2_genes]
        covid_sum = sum_df[scov2_genes]
        spot_means.append(covid_frac.max(axis=1).values)
        # [gene_means.append(x) for x in covid_frac.mean(axis=0).values]

        fractions.append(covid_frac.values.flatten())
        counts.append((covid_sum * covid_frac).values.flatten())
        spot_fractions.append(covid_frac)

    spot_means = np.vstack(spot_means).max(axis=0)
    spot_means = spot_means.reshape(-1, 1)
    return spot_means


def decontaminate(adata, total_counts_df, contamination):
    contamination_counts = total_counts_df * contamination
    observed_expr = adata.X.toarray()
    observed_expr -= contamination_counts
    observed_expr = observed_expr.astype(int)
    observed_expr = np.maximum(observed_expr, 0)
    adata.X = csr_matrix(observed_expr)


sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
samples.sort(key=len)

metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)
# samples from the first batch
selected_samples = list(metadata_df[metadata_df['batch'] == '2022/11/01'].index)
mock_adatas = ['L221201', 'L221202']  # two mock treated samples

scov2_genes = [
    'scov2_gene-GU280_gp01',
    'scov2_gene-GU280_gp02',
    'scov2_gene-GU280_gp03',
    'scov2_gene-GU280_gp04',
    'scov2_gene-GU280_gp05',
    'scov2_gene-GU280_gp10',
    # 'scov2_gene-GU280_gp11',  # not expressed
]

#%%
# plots for sanity check
sample_names = [p.split('/')[-1].split('_')[0] for p in samples]

# average based
sample_adatas = {k: sc.read_h5ad(v) for k, v in zip(sample_names, samples) if k in selected_samples}

plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'scov2_gene-GU280_gp05')
plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'ENSMUSG00000017300')

cont_fraction = calculate_avg_contamination_fraction(sample_adatas, mock_adatas, scov2_genes)
total_counts_df = get_sum_counts(list(sample_adatas.values()))

for label, sample in sample_adatas.items():
    decontaminate(sample, total_counts_df, cont_fraction)

plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'scov2_gene-GU280_gp05')
plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'ENSMUSG00000017300')

# per spot based
sample_adatas = {k: sc.read_h5ad(v) for k, v in zip(sample_names, samples) if k in selected_samples}

cont_fraction = calculate_contamination_fraction(sample_adatas, mock_adatas, scov2_genes)
total_counts_df = get_sum_counts(list(sample_adatas.values()))

for label, sample in sample_adatas.items():
    decontaminate(sample, total_counts_df, cont_fraction)

cont_fraction_df = pd.DataFrame(cont_fraction)

plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'scov2_gene-GU280_gp10')
plot_spatial_histology_and_gene_expression(list(sample_adatas.values()), 'ENSMUSG00000017300')

#%%
# actually perform the removal and save the samples

sample_names = [p.split('/')[-1].split('_')[0] for p in samples]
sample_adatas = {k: sc.read_h5ad(v) for k, v in zip(sample_names, samples) if k in selected_samples}

cont_fraction = calculate_contamination_fraction(sample_adatas, mock_adatas, scov2_genes)
total_counts_df = get_sum_counts(list(sample_adatas.values()))

for label, sample in tqdm(sample_adatas.items()):
    decontaminate(sample, total_counts_df, cont_fraction)
    sample.write(f'{sample_folder}/{label}_{sample_suffix}.h5ad')
