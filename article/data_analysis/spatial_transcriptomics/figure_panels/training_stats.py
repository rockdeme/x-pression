import numpy as np
import pandas as pd
from glob import glob
import scanpy as sc
import chrysalis as ch
import matplotlib.pyplot as plt
import json


file_dir = '/mnt/f/3d_tomography/cnn_training/output/results_v3/'
experiment = 'single_sample_v3_75epoch_64-64-21'

labels_df = pd.DataFrame()
csvs = glob(file_dir + experiment + '_predictions*.csv')

for f in csvs:
    split = f.split('_')[-1][:-4].lower()
    df = pd.read_csv(f, index_col=0)
    if split == 'val':
        df['split'] = 'validation'
    else:
        df['split'] = split
    labels_df = pd.concat([labels_df, df], ignore_index=True)

labels_df = labels_df.drop_duplicates(subset=['barcode'])
labels_df.index = labels_df['barcode']

print(labels_df['split'].value_counts())

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')
hexcodes = ch.utils.get_hexcodes(None, 8, 42, len(adata))

compartments_df = pd.DataFrame(adata.obsm['chr_aa'])
compartments_df = compartments_df.idxmax(axis=1)
adata = adata[adata.obs.index.isin(list(labels_df['barcode']))]
adata.obs['split'] = labels_df['split']
adata.obs['class'] = list(compartments_df)
adata.obs['class'] = adata.obs['class'].astype('category')

annot_dict = {'atelectasis': 'Atelectasis', 'inflammation': 'Inflammation',
              'lung_parenchyma': 'Lung parenchyma', 'unlabeled': 'Unlabeled'}
adata.obs['annotation'] = [annot_dict[x] for x in adata.obs['annotation']]

sc.pl.spatial(adata, color=['split'], s=12)
sc.pl.spatial(adata, color=['class'], s=12, palette=hexcodes, alpha_img=0.7)

# spatial plots
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sc.set_figure_params(vector_friendly=True, dpi_save=200)
sc.pl.spatial(adata, color=['class'], s=12, palette=hexcodes, alpha_img=0.7, ax=axes[0], show=False)
axes[0].set_xlabel('')
axes[0].set_ylabel('')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].set_title('Compartment classes')
leg = axes[0].get_legend()
if leg is not None:
    leg.set_title("Class id")
sc.set_figure_params(vector_friendly=True, dpi_save=200)
sc.pl.spatial(adata, color=['split'], s=12, palette=['#f04242', '#d086bc', '#42f0a7'],
              alpha_img=0.7, ax=axes[1], show=False)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].set_title('Dataset splits')
leg = axes[1].get_legend()
if leg is not None:
    leg.set_title("Set")
sc.set_figure_params(vector_friendly=True, dpi_save=200)
sc.pl.spatial(adata, color=['annotation'], s=12, palette=['#dec354', '#de5474', '#7b4fe3', '#9084ae'],
              alpha_img=0.7, ax=axes[2], show=False)
axes[2].set_xlabel('')
axes[2].set_ylabel('')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['left'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].set_title('Pathologist annotations')
leg = axes[2].get_legend()
if leg is not None:
    leg.set_title("Annotation")

plt.tight_layout()
plt.savefig('data/figures/training_spatial_plots.svg')
plt.show()

# training curves
experiment = 'single_sample_v3_75epoch_64-64-21'
with open(file_dir + f'{experiment}_history.json', 'r') as f:
    train_dict = json.load(f)

train_loss = train_dict['loss']
val_loss = train_dict['val_loss']
train_mae = train_dict['mae']
val_mae = train_dict['val_mae']

fig, axes = plt.subplots(2, 1, figsize=(3.75, 6.25))

axes[0].plot(train_loss, label='Training loss', color='#d086bc', linewidth=2)
axes[0].plot(val_loss, label='Validation loss', color='#42f0a7', linewidth=2)
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].grid(True, zorder=0)
axes[0].legend()

axes[1].plot(train_mae, label='Training MAE', color='#d086bc', linewidth=2)
axes[1].plot(val_mae, label='Validation MAE', color='#42f0a7', linewidth=2)
axes[1].set_title('MAE over epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('MAE')
axes[1].grid(True, zorder=0)
axes[1].legend()

plt.tight_layout()
plt.savefig('data/figures/training_curves.svg')
plt.show()
