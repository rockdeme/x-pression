import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc


def write_tiff_original_scale(input_file, output_file, num_channels=9, scaling_factor=32,
                              x_clip=None, y_clip=None, z_clip=None, convert_to_int=False):
    with tiff.TiffFile(input_file) as tif:
        height, width = tif.pages[0].shape
        depth = len(tif.pages)
        Z = depth // num_channels

        if x_clip:
            width = x_clip[1] - x_clip[0]
        if y_clip:
            height = y_clip[1] - y_clip[0]

        with tiff.TiffWriter(output_file, bigtiff=True) as tif_writer:
            for i in tqdm(range(Z), desc='Writing slices to TIFF...'):
                slice_factor = i * num_channels
                slices = np.stack([page.asarray() for page in
                                   tif.pages[slice_factor:slice_factor + num_channels]], axis=0)
                slices = slices.astype(np.float32)

                if x_clip:
                    slices = slices[:, :, x_clip[0]:x_clip[1]]
                if y_clip:
                    slices = slices[:, y_clip[0]:y_clip[1], :]

                y_transpose = slices[0].shape[0] % scaling_factor
                x_transpose = slices[0].shape[1] % scaling_factor

                assert (slices[0].shape[0] - y_transpose) % scaling_factor == 0, 'Y transpose is invalid'
                assert (slices[0].shape[1] - x_transpose) % scaling_factor == 0, 'X transpose is invalid'

                h, w = slices.shape[1:3]

                slices = slices[:, y_transpose:, x_transpose:]

                downsampled_slices = []
                for s in slices:
                    # reshape and take the top-left pixel of each 32x32 block
                    ds = s.reshape(h // scaling_factor, scaling_factor, w // scaling_factor,
                                   scaling_factor)[:, 0, :, 0]
                    downsampled_slices.append(ds)
                downsampled_slices = np.stack(downsampled_slices)

                if convert_to_int:
                    downsampled_slices = (downsampled_slices * 255).astype(np.uint8)
                for slice_img in downsampled_slices:
                    tif_writer.write(slice_img, photometric='minisblack')


#%%
# save the predictions as tiff at original scale
# input_file = '/mnt/d/data/covid/tomography/prediction/combined_slices_depth11_stride32_0and1.tiff'
input_file = '/mnt/f/3d_tomography/cnn_training/combined_slices_depth11_stride32_0and1.tiff'
x_clip = [1416, 10656]
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_original_scale.tiff"
write_tiff_original_scale(input_file, save_file, num_channels=9, scaling_factor=32, x_clip=x_clip,
                          convert_to_int=False)

input_file = '/mnt/f/3d_tomography/cnn_training/L2210914/L2210914.tiff'
x_clip = [250, 9000]
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_original_scale.tiff"
write_tiff_original_scale(input_file, save_file, num_channels=9, scaling_factor=32, x_clip=x_clip,
                          convert_to_int=False)

#%%
num_channels = 9
colormaps = ['#dbc257', '#5770db', '#db5f57', '#db57b2', '#91db57', '#57d3db', '#57db80', '#a157db']

# mRNA sample
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_downscaled_original_scale.tiff"
with tiff.TiffFile(save_file) as tif:
    num_pages = len(tif.pages)
    height, width = tif.pages[0].shape
    Z = num_pages // num_channels  # Number of Z slices

    # Read and stack slices into a (Z, C, H, W) array
    image_4d_mrna = np.stack([
        np.stack([tif.pages[i * num_channels + c].asarray() for c in range(num_channels)], axis=0)
        for i in range(Z)
    ], axis=0)

props_array = []
for s in image_4d_mrna:
    channels = s[1:, :, :]
    props = channels.sum(axis=1).sum(axis=1)
    tissue = channels.sum(axis=0) != 0  # (Z, C, W, H)
    n_pixels = tissue.sum()
    props = props / n_pixels
    props_array.append(props)

props_array = np.stack(props_array)
mrna_props_df = pd.DataFrame(props_array, index=[x * 1.6 * 21 for x in range(props_array.shape[0])])

# OTS sample
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_original_scale.tiff"
with tiff.TiffFile(save_file) as tif:
    num_pages = len(tif.pages)
    height, width = tif.pages[0].shape
    Z = num_pages // num_channels  # Number of Z slices

    # Read and stack slices into a (Z, C, H, W) array
    image_4d_ots = np.stack([
        np.stack([tif.pages[i * num_channels + c].asarray() for c in range(num_channels)], axis=0)
        for i in range(Z)
    ], axis=0)

props_array = []
for s in image_4d_ots:
    channels = s[1:, :, :]
    props = channels.sum(axis=1).sum(axis=1)
    tissue = channels.sum(axis=0) != 0  # (Z, C, W, H)
    n_pixels = tissue.sum()
    props = props / n_pixels
    props_array.append(props)
props_array = np.stack(props_array)
ots_props_df = pd.DataFrame(props_array, index=[x * 1.6 * 21 for x in range(props_array.shape[0])])

#%%
# area plot

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')

comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)

sum_df = comps_df.sum(axis=0) / comps_df.sum(axis=0).sum()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 3, figsize=(6, 3.5), gridspec_kw={'width_ratios': [1, 6, 6]})

bottoms = np.zeros(len(sum_df))
for i, (val, color) in enumerate(zip(sum_df, colormaps)):
    axes[0].bar(1, val, bottom=bottoms[i], color=color, alpha=0.8)
    bottoms[i + 1:] += val
for b in bottoms[1:]:  # Skip the bottom-most (0)
    axes[0].hlines(y=b, xmin=0.7, xmax=1.3, color='black', linewidth=1)
axes[0].set_xlim(0.8, 1.0)
axes[0].set_ylim(0, 1)
axes[0].set_xticks([])
axes[0].set_ylabel("Proportion")
axes[0].set_title("2D")

x = mrna_props_df.index
y = mrna_props_df.values.T
axes[1].stackplot(x, y, labels=[f'{i}' for i in range(mrna_props_df.shape[1])], colors=colormaps, alpha=0.8)
axes[1].set_xlim(x.min(), x.max())
axes[1].set_ylim(0, 1)
box = axes[1].get_position()
axes[1].set_position([box.x0, box.y0, box.width * 0.95, box.height])
for i in range(1, mrna_props_df.shape[1]):
    axes[1].plot(x, np.cumsum(mrna_props_df.values[:, :i], axis=1).T[-1], color='black', lw=1)
axes[1].set_title("mRNA vaccine")

x = ots_props_df.index
y = ots_props_df.values.T
axes[2].stackplot(x, y, labels=[f'{i}' for i in range(ots_props_df.shape[1])], colors=colormaps, alpha=0.8)
axes[2].set_xlim(x.min(), x.max())
axes[2].set_ylim(0, 1)
box = axes[2].get_position()
axes[2].set_position([box.x0, box.y0, box.width * 0.95, box.height])
for i in range(1, ots_props_df.shape[1]):
    axes[2].plot(x, np.cumsum(ots_props_df.values[:, :i], axis=1).T[-1], color='black', lw=1)
axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Components", frameon=False)
axes[2].set_xlabel("Depth (μm)")
axes[2].set_ylabel("Proportion")
axes[2].set_title("OTS206")

plt.suptitle("Tissue composition")
plt.tight_layout()
# plt.savefig('data/figures/3d_tissue_composition.svg')
plt.show()

# %%
num_channels = 9
colormaps = ['#dbc257', '#5770db', '#db5f57', '#db57b2', '#91db57', '#57d3db', '#57db80', '#a157db']

# mRNA sample
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_downscaled_original_scale.tiff"
with tiff.TiffFile(save_file) as tif:
    num_pages = len(tif.pages)
    height, width = tif.pages[0].shape
    Z = num_pages // num_channels  # Number of Z slices

    # Read and stack slices into a (Z, C, H, W) array
    image_4d_mrna = np.stack([
        np.stack([tif.pages[i * num_channels + c].asarray() for c in range(num_channels)], axis=0)
        for i in range(Z)
    ], axis=0)

array = []
for s in image_4d_mrna:
    channels = s[1:, :, :]
    tissue = channels.sum(axis=0) != 0  # (Z, C, W, H)

    channels_vec = channels[:, tissue].T
    channel_max = pd.Series(np.argmax(channels_vec, axis=1))
    channel_max = channel_max.value_counts() / channel_max.value_counts().sum()
    channel_max = channel_max.sort_index()
    channel_max = channel_max.reindex(range(channels.shape[0]), fill_value=0)
    array.append(channel_max.values)

array = np.stack(array)
mrna_props_df = pd.DataFrame(array, index=[x * 1.6 * 21 for x in range(array.shape[0])])

# OTS sample
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_original_scale.tiff"
with tiff.TiffFile(save_file) as tif:
    num_pages = len(tif.pages)
    height, width = tif.pages[0].shape
    Z = num_pages // num_channels  # Number of Z slices

    # Read and stack slices into a (Z, C, H, W) array
    image_4d_ots = np.stack([
        np.stack([tif.pages[i * num_channels + c].asarray() for c in range(num_channels)], axis=0)
        for i in range(Z)
    ], axis=0)

array = []
for s in image_4d_ots:
    channels = s[1:, :, :]
    tissue = channels.sum(axis=0) != 0  # (Z, C, W, H)

    channels_vec = channels[:, tissue].T
    channel_max = pd.Series(np.argmax(channels_vec, axis=1))
    channel_max = channel_max.value_counts() / channel_max.value_counts().sum()
    channel_max = channel_max.sort_index()
    channel_max = channel_max.reindex(range(channels.shape[0]), fill_value=0)
    array.append(channel_max.values)

array = np.stack(array)
ots_props_df = pd.DataFrame(array, index=[x * 1.6 * 21 for x in range(array.shape[0])])

# %%
# area plot

adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')

comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)

sum_df = comps_df.idxmax(axis=1).value_counts() / comps_df.shape[0]
sum_df = sum_df.sort_index()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 3, figsize=(6, 3.5), gridspec_kw={'width_ratios': [1, 6, 6]})

bottoms = np.zeros(len(sum_df))
for i, (val, color) in enumerate(zip(sum_df, colormaps)):
    axes[0].bar(1, val, bottom=bottoms[i], color=color, alpha=0.8)
    bottoms[i + 1:] += val
for b in bottoms[1:]:  # Skip the bottom-most (0)
    axes[0].hlines(y=b, xmin=0.7, xmax=1.3, color='black', linewidth=1)
axes[0].set_xlim(0.8, 1.0)
axes[0].set_ylim(0, 1)
axes[0].set_xticks([])
axes[0].set_ylabel("Proportion")
axes[0].set_title("2D")

x = mrna_props_df.index
y = mrna_props_df.values.T
axes[1].stackplot(x, y, labels=[f'{i}' for i in range(mrna_props_df.shape[1])], colors=colormaps, alpha=0.8)
axes[1].set_xlim(x.min(), x.max())
axes[1].set_ylim(0, 1)
box = axes[1].get_position()
axes[1].set_position([box.x0, box.y0, box.width * 0.95, box.height])
for i in range(1, mrna_props_df.shape[1]):
    axes[1].plot(x, np.cumsum(mrna_props_df.values[:, :i], axis=1).T[-1], color='black', lw=1)
axes[1].set_title("mRNA vaccine")

x = ots_props_df.index
y = ots_props_df.values.T
axes[2].stackplot(x, y, labels=[f'{i}' for i in range(ots_props_df.shape[1])], colors=colormaps, alpha=0.8)
axes[2].set_xlim(x.min(), x.max())
axes[2].set_ylim(0, 1)
box = axes[2].get_position()
axes[2].set_position([box.x0, box.y0, box.width * 0.95, box.height])
for i in range(1, ots_props_df.shape[1]):
    axes[2].plot(x, np.cumsum(ots_props_df.values[:, :i], axis=1).T[-1], color='black', lw=1)
axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Components", frameon=False)
axes[2].set_xlabel("Depth (μm)")
axes[2].set_title("OTS206")

plt.suptitle("Tissue composition")
plt.tight_layout()
plt.savefig('data/figures/3d_tissue_composition_argmax.svg')
plt.show()

#%%

# distributions
# boxplots

def process_image_channels(image_4d):
    channels_4d = image_4d[:, 1:].copy()
    channels_4d = np.swapaxes(channels_4d, 0, 1)

    tissue = channels_4d.sum(axis=0) != 0  # (Z, C, W, H)
    bg = channels_4d.sum(axis=0) == 0  # (Z, C, W, H)

    channels_4d[:, bg] = np.nan
    channels_4d = channels_4d.reshape(channels_4d.shape[0], -1)

    df = pd.DataFrame(channels_4d.T)
    df = df.dropna(axis="rows")

    return df

# Apply function to both datasets
channels_mrna_df = process_image_channels(image_4d_mrna)
channels_ots_df = process_image_channels(image_4d_ots)


fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))

sns.boxplot(comps_df, ax=axs[0], palette=colormaps, showfliers=False)
axs[0].axis('off')
axs[0].axis('on')
axs[0].set_ylim(-0.05, 1.05)
axs[0].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
axs[0].set_axisbelow(True)
axs[0].set_ylabel('Value per spot')
axs[0].set_title('2D ST section', fontsize=10)
axs[0].set_xlabel(None)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

sns.boxplot(channels_mrna_df, ax=axs[1], palette=colormaps, showfliers=False)
axs[1].axis('off')
axs[1].axis('on')
axs[1].set_ylim(-0.05, 1.05)
axs[1].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
axs[1].set_axisbelow(True)
axs[1].set_ylabel('Predicted value per voxel')
axs[1].set_title('3D predictions', fontsize=10)
axs[1].set_xlabel(None)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
fig.supylabel(None)

sns.boxplot(channels_ots_df, ax=axs[2], palette=colormaps, showfliers=False)
axs[2].axis('off')
axs[2].axis('on')
axs[2].set_ylim(-0.05, 1.05)
axs[2].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
axs[2].set_axisbelow(True)
axs[2].set_ylabel('Predicted value per voxel')
axs[2].set_title('3D predictions', fontsize=10)
axs[2].set_xlabel(None)
axs[2].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
fig.supylabel(None)

plt.tight_layout()
# plt.savefig('data/figures/spot_distribution_2d_3d.svg')
plt.show()

# compartment distributions
# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(8, 4))
axs = axs.flatten()
for idx, col in enumerate(comps_df.columns):

    # subsample for testing plots
    n_sample = 10000
    rna_vec = np.random.choice(channels_mrna_df[idx], n_sample, replace=False)
    ots_vec = np.random.choice(channels_ots_df[idx], n_sample, replace=False)
    sns.histplot(rna_vec, ax=axs[idx], kde=True, bins=30, alpha=0.5, label='DF1', binrange=(0, 1))
    sns.histplot(ots_vec, ax=axs[idx], kde=True, bins=30, alpha=0.5, label='DF2', binrange=(0, 1))
    axs[idx].set_xlim(0, 1)

    # sns.kdeplot(channels_mrna_df[idx], ax=axs[idx], fill=True, alpha=0.5, label='DF1')
    # sns.kdeplot(channels_ots_df[idx], ax=axs[idx], fill=True, alpha=0.5, label='DF2')
    # axs[idx].hist(comps_df[col], bins=30, density=True, alpha=0.3, label="Dataset 1", color="blue")
    # axs[idx].hist(channels_mrna_df[idx], bins=30, density=True, alpha=0.3, label="Dataset 2", color="orange")
    axs[idx].set_title(col)
    axs[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    axs[idx].set_axisbelow(True)
    axs[idx].set_ylabel('Density')
    axs[idx].set_xlabel(None)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=0)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
    # axs[idx].set_xlim(0, 1)
plt.suptitle('Compartment distributions')
plt.tight_layout()
plt.show()

#%%
# Define fixed subplot grid (4 rows, 4 columns)
from matplotlib.ticker import FuncFormatter

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

def scientific_notation(x, pos):
    return '{:.0e}'.format(x).replace('e+0', 'e')


fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(2, 1, hspace=0.25)  # spacing between the two groups

gs0 = gs[0].subgridspec(2, 4, hspace=0.1, wspace=0.8, height_ratios=[2, 1])
gs1 = gs[1].subgridspec(2, 4, hspace=0.1, wspace=0.8, height_ratios=[2, 1])


# Pre-allocate subplots
set1, set1_sub = gs0.subplots()  # 2x4 grid (top half)
set2, set2_sub = gs1.subplots()  # 2x4 grid (bottom half)

palette = ['#a866ff', '#ff668a']
df1_color = palette[0]  # Color for DF1
df2_color = palette[1]  # Color for DF2

# Loop through each column and assign the correct subplot
for idx, col in enumerate(comps_df.columns):
    if idx < 4:
        ax_kde = set1[idx]  # KDE plot
        ax_box = set1_sub[idx]  # Boxplot
    else:
        idx -= 4
        ax_kde = set2[idx]  # KDE plot (lower row)
        ax_box = set2_sub[idx]  # Boxplot (lower row)

    # Subsample for testing
    n_sample = min(int(1e6), len(channels_mrna_df[col]), len(channels_ots_df[col]))
    rna_vec = np.random.choice(channels_mrna_df[col], n_sample, replace=False)
    ots_vec = np.random.choice(channels_ots_df[col], n_sample, replace=False)

    # rna_vec = channels_mrna_df[col]
    # ots_vec = channels_ots_df[col]

    # KDE plots
    # sns.kdeplot(rna_vec, ax=ax_kde, fill=True, alpha=0.5, label='DF1')
    # sns.kdeplot(ots_vec, ax=ax_kde, fill=True, alpha=0.5, label='DF2')
    sns.histplot(rna_vec, ax=ax_kde, kde=False, bins=30, alpha=0.5, label='DF1', binrange=(0, 1), color=df1_color)
    sns.histplot(ots_vec, ax=ax_kde, kde=False, bins=30, alpha=0.5, label='DF2', binrange=(0, 1), color=df2_color)

    ax_kde.set_title(col)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.grid(axis='y', linestyle='-', linewidth=0.5, color='grey')
    # ax_kde.set_xticks([0, 1])
    ax_kde.set_xlim([0, 1])
    ax_kde.set_axisbelow(True)
    ax_kde.set_ylabel('Counts')
    ax_kde.set_xticklabels([])
    ax_kde.set_facecolor("#f2f2f2")
    ax_kde.yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    # Boxplots below KDE
    sns.boxplot(data=[rna_vec, ots_vec], ax=ax_box, orient='h', showfliers=False, widths=0.85,
                palette=[df1_color, df2_color], boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
                medianprops=dict(color='black'), capprops=dict(color='black'))
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(['0', '1'])
    ax_box.set_ylabel(None)
    ax_box.grid(False)
    ax_box.set_xlim([0, 1])
    ax_box.axis('off')
    # ax_box.set_xlabel('Value per spot')

handles = [
    plt.Line2D([0], [0], color=df1_color, lw=4, label="mRNA"),
    plt.Line2D([0], [0], color=df2_color, lw=4, label="OTS206")
]
fig.legend(handles=handles, loc='center right', title="Condition", frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5, top=0.85, right=0.825)
plt.suptitle('Compartment distributions')
plt.savefig('data/figures/3d_distributions.svg')
# plt.tight_layout()
plt.show()
