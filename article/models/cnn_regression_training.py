import os
import gc
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras import layers, models
from tqdm import tqdm
import dask.array as da
import math
import zarr
import random
from glob import glob
import json

import keras
import tensorflow as tf

# UPDATE!
parent_dir = '/mnt/c/Users/Deme/PycharmProjects/CovidParty'
sys.path.append(parent_dir)

INPUT_H5AD = '/mnt/d/data/covid/tomography/adata_spatial_transformed_4.h5ad'
INPUT_TIFF = '/mnt/d/data/covid/tomography/rotated_full_volume2.tiff'
COMPARTMENT_CSV = '/mnt/d/data/covid/tomography/chr_comps_5_combined_0-1.csv'
OUTPUT_TIFF = '/mnt/d/data/covid/tomography/slices_subset.tiff'
OUTPUT_ZARR = '/mnt/d/data/covid/tomography/slices.zarr'
OUTPUT_PATCH = '/mnt/d/data/covid/tomography/output/patches_v3/patches.zarr'

INPUT_FOLDER = '/mnt/d/data/covid/tomography/input/'
OUTPUT_FOLDER = '/mnt/d/data/covid/tomography/output/'
SAVE_DIR = '/mnt/d/data/covid/tomography/output/results_v3/'

# SAVE_DIR = '/mnt/f/3d_tomography/data/'
# INPUT_H5AD = '/mnt/f/3d_tomography/adata_spatial_transformed_4.h5ad'
# INPUT_TIFF = '/mnt/f/3d_tomography/rotated_full_volume2.tiff'
# COMPARTMENT_CSV = '/mnt/f/3d_tomography/chr_comps_5_combined_0-1.csv'
# OUTPUT_TIFF = '/mnt/f/3d_tomography/data/slices_subset.tiff'
# OUTPUT_ZARR = '/mnt/f/3d_tomography/data/slices.zarr'
# OUTPUT_PATCH = '/mnt/f/3d_tomography/data/patches.zarr'

CENTER_SLICES = {'L22_109_26': 296, 'L22_75_12': 623, 'L22_109_6': 502}
CENTER_SLICE = 296
NUM_SLICES = 10 # depth: 1+n*2
WIDTH_HEIGHT = 64 # radius 64*2=128

# n_slices, width_height_px = 5, 16  # this is just for testing
n_slices, width_height_px = NUM_SLICES, WIDTH_HEIGHT

def read_adata(file_path):
    try:
        adata = sc.read(file_path)
        return adata
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        raise e

def calculate_min_max(input_file):
    """Calculate the global min and max values over all slices."""
    # we use the inf trick to make any real value larger or smaller and trigger the first update
    global_min = float('inf')
    global_max = float('-inf')

    with tiff.TiffFile(input_file) as tif:
        n_slices = len(tif.pages)
        for i in tqdm(range(n_slices), desc='Grabbing min and max values...'):
            slice_data = tif.pages[i].asarray()
            global_min = min(global_min, slice_data.min())
            global_max = max(global_max, slice_data.max())
    return global_min, global_max

def extract_and_save_slices(input_file, center_slice, output_file, num_slices):
    """Extract and save subset of specified depth."""
    start_index = center_slice - num_slices
    end_index = center_slice + num_slices
    
    with tiff.TiffFile(input_file) as tif:
        slices = [tif.pages[i].asarray() for i in range(start_index, end_index + 1)]
    
    tiff.imwrite(output_file, slices, bigtiff=True)
    print('Slices extracted.')

def extract_and_save_slices_to_zarr(input_file, output_directory, center_slice, num_slices, scale=True):
    """Extract and save a subset of specified depth as a Zarr dataset."""
    start_index = center_slice - num_slices
    end_index = center_slice + num_slices

    with tiff.TiffFile(input_file) as tif:
        first_slice = tif.pages[start_index].asarray()
        height, width = first_slice.shape
        zarr_array = zarr.create(shape=(end_index - start_index + 1, height, width),
                                 chunks=(1, height, width),
                                 dtype=float,
                                 store=output_directory,
                                 overwrite=True)

        for i in tqdm(range(start_index, end_index + 1), 'Writing slices to zarr...'):
            slice_data = tif.pages[i].asarray()
            if scale:
                slice_data = slice_data.astype(float)
                slice_data /= 255.0
            zarr_array[i - start_index] = slice_data

def get_patches3d(spatial_coords, volume, radius, z_radius):
    """Extract 3D patches from the volume"""
    patches = []
    depth, height, width = volume.shape
    
    for point in tqdm(spatial_coords, desc='Grabbing patches...'):
        x, y, z = point
        top_left = (x - radius, y - radius, z - z_radius)
        bottom_right = (x + radius, y + radius, z + z_radius)
        
        x1, y1, z1 = map(int, top_left)
        x2, y2, z2 = map(int, bottom_right)
        
        # Bound checking
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        z1, z2 = max(0, z1), min(depth, z2)
        
        patch = volume[z1:z2 + 1, y1:y2, x1:x2]
        patches.append(np.array(patch))
    return np.array(patches)

def get_patches3d_dask(spatial_coords, volume, radius, z_radius):
    """Extract 3D patches from the volume using dask arrays"""
    patches = []
    depth, height, width = volume.shape

    for point in spatial_coords:
        x, y, z = point
        top_left = (x - radius, y - radius, z - z_radius)
        bottom_right = (x + radius, y + radius, z + z_radius)

        x1, y1, z1 = map(int, top_left)
        x2, y2, z2 = map(int, bottom_right)

        # Bound checking
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        z1, z2 = max(0, z1), min(depth, z2)

        patch = volume[z1:z2 + 1, y1:y2, x1:x2]
        patches.append(patch)
    return da.stack(patches).compute()

def save_patches_to_zarr(patches, output_directory, overwrite=False):
    """Save 3D patches as a Zarr dataset."""
    n_patches, depth, height, width, channel = patches.shape
    if overwrite is True:
        zarr_array = zarr.create(shape=(n_patches, depth, height, width, channel),
                                 chunks=(1, depth, height, width, channel),  # chunking per patch
                                 dtype=patches.dtype,
                                 store=output_directory,
                                 overwrite=True)
        zarr_array[:] = patches
    else:
        if os.path.exists(output_directory):
            zarr_array = zarr.open(output_directory, mode='r+')
            current_patches = zarr_array.shape[0]
            zarr_array.resize((current_patches + n_patches, depth, height, width, channel))
            zarr_array[current_patches:current_patches + n_patches, :, :, :, :] = patches
        else:
            zarr_array = zarr.create(shape=(n_patches, depth, height, width, channel),
                                     chunks=(1, depth, height, width, channel),  # chunking per patch
                                     dtype=patches.dtype,
                                     store=output_directory,
                                     overwrite=False)
            zarr_array[:] = patches

    print(f'Patches saved to {output_directory}')

def load_patch_and_labels(zarr_path, indices, labels, batch_size, d, w, h,
                          num_labels, buffer_size=1000):
    """Load patches and corresponding labels in batches based on provided indices."""
    zarr_array = zarr.open(zarr_path, mode='r')

    def generator():
        for i in indices:
            patch = zarr_array[i]
            # patch = tf.convert_to_tensor(patch)
            # patch.set_shape((d, w, h, 1))
            yield patch, labels[i]

    # this is just some debugging
    # for patch, label in generator():
    #     print(f'Patch shape: {patch.shape}, Label: {label}')

    # create a tensorflow dataset from the generator
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(d, w, h, 1), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(num_labels,), dtype=tf.float32)
                                             ))
    # shuffle and batch the dataset
    # todo: try stuff with batch -> repeat
    dataset = dataset.shuffle(buffer_size=buffer_size).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_patch_and_labels_no_shuffle(zarr_path, indices, labels, batch_size, d, w, h, num_labels):
    zarr_array = zarr.open(zarr_path, mode='r')
    def generator():
        for i in indices:
            patch = zarr_array[i]
            yield patch, labels[i]
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(d, w, h, 1), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(num_labels,), dtype=tf.float32)
                                             ))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def kl_divergence(y_true, y_pred):
    # Clip to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)
    y_true = tf.clip_by_value(y_true, tf.keras.backend.epsilon(), 1.0)
    return tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=1))

def create_3d_model_single_output(input_shape):
    keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv3D(32, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(256, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
    return model

def create_3d_model_multi_output(input_shape, n_output=8):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv3D(32, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(256, (3, 9, 9), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_output, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=kl_divergence, metrics=['mae'])
    return model

def create_feature_model_3d_multi_output(input_shape):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Dense(256, activation='relu'))   
    model.add(layers.Dropout(0.4))                    

    model.add(layers.Dense(128, activation='relu'))   
    model.add(layers.Dropout(0.3))                   

    model.add(layers.Dense(64, activation='relu'))    
    model.add(layers.Dropout(0.2))                    

    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=kl_divergence, metrics=['mae'])
    
    return model

def plot_training_history(history, save_path):
    """Save the training history plot to a file."""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_loss, label='Training Loss (MSE)', color='blue')
    axes[0].plot(val_loss, label='Validation Loss (MSE)', color='orange')
    axes[0].set_title('Loss (Mean Squared Error) Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()

    axes[1].plot(train_mae, label='Training MAE', color='blue')
    axes[1].plot(val_mae, label='Validation MAE', color='orange')
    axes[1].set_title('Mean Absolute Error (MAE) Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spatial(adata, path, y_true, y_pred):
    n_cols = 2
    n_rows = y_true.shape[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))

    for i in range(n_rows):
        adata.obs[f'{i}_truth'] = y_true[:, i]
        adata.obs[f'{i}_pred'] = y_pred[:, i]
        sc.pl.spatial(adata, color=f'{i}_truth', s=12, ax=axes[i, 0], show=False)
        axes[i, 0].set_title(f'{i} true values')
        sc.pl.spatial(adata, color=f'{i}_pred', s=12, ax=axes[i, 1], show=False)
        axes[i, 1].set_title(f'{i} predictions')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_first_slices_with_titles(batch):
    images, labels = batch
    batch_size = images.shape[0]  # 16 in your case
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))

    for i in range(batch_size):
        # Extract the first slice (shape: 64x64)
        first_slice = images[i, 0, :, :, 0]

        # Plot the first slice
        axes[i].imshow(first_slice, cmap='gray')

        # Add title (label)
        axes[i].set_title(f"Label: {labels[i].numpy().round(2)}")

        # Hide axis for cleaner look
        axes[i].axis('off')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def bin_examples(y, n_bins):
    if y.nunique() == 1:  # If all values are the same, assign them to bin 0
        return pd.Series(0, index=y.index, dtype=int)

    bins = np.linspace(y.min(), y.max(), num=n_bins + 1)
    y_bins = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    bin_counts = np.bincount(y_bins)
    most_populated_bin = np.argmax(bin_counts)
    return y_bins

def bin_multilabel(df, n_bins):
    df_dict = {}
    for c in df.columns:
        y_bins = bin_examples(df[c], n_bins)
        df_dict[f'{c}_bin'] = y_bins
    return pd.DataFrame(df_dict)

#%%
# get input files
input_folders = glob(INPUT_FOLDER + '*/')
input_dict = {}
for f in input_folders:
    fname = f.split('/')[-2]
    assert fname in CENTER_SLICES.keys(), 'Volume center slice is not specified in CENTER_SLICES.'
    input_dict[fname] = {
        'center_slice': CENTER_SLICES[fname],
        'tiff': glob(f + 'rotated.tiff')[0],
        'adata': glob(f + '*.h5ad')[0],
        'compartment_csv': glob(f + '*.csv')[0],
    }

#%%
# save slices from volumes to zarr
for x in list(input_dict.keys()):
    output_zarr = OUTPUT_FOLDER + x + '.zarr'
    # we do the scaling inside this function now
    extract_and_save_slices_to_zarr(input_dict[x]['tiff'], output_zarr, input_dict[x]['center_slice'], n_slices)

#%%
# stratify, bin, and save patches in batches
split_counter = 0
for x in list(input_dict.keys()):
    output_zarr = OUTPUT_FOLDER + x + '.zarr'
    # we do the scaling inside this function now

    tomography_volume = da.from_zarr(output_zarr)
    adata = read_adata(input_dict[x]['adata'])

    # grab y and spatial locs for patches
    compartment_df = pd.read_csv(input_dict[x]['compartment_csv'], index_col=0)
    adata.obs['filter'] = compartment_df[compartment_df.columns[0]]
    adata = adata[~adata.obs['filter'].isna()]

    # lul
    if 'spatial_transformed' in adata.obsm.keys():
        spatial_coords = adata.obsm['spatial_transformed']
    elif 'transformed_spatial' in adata.obsm.keys():
        spatial_coords = adata.obsm['transformed_spatial']

    # remove spots that are too close to the boundary
    out_of_bounds = np.any(0 >= spatial_coords - width_height_px, axis=1)
    out_of_bounds = np.where(out_of_bounds)[0]
    out_of_bounds_labels = adata.obs.index[list(out_of_bounds)]
    adata = adata[~adata.obs.index.isin(out_of_bounds_labels)]
    compartment_df = compartment_df.loc[adata.obs_names]
    spatial_coords = np.delete(spatial_coords, out_of_bounds, axis=0)


    depth, height, width = tomography_volume.shape
    z_slice = depth // 2
    spatial_coords_3d = np.hstack([spatial_coords, np.full((spatial_coords.shape[0], 1), z_slice)])

    # get labels
    # y = np.array(adata.obs['0'])
    # y_min, y_max = y.min(), y.max ()
    # y_normalized = (y - y_min) / (y_max - y_min)

    # stratified sampling
    # bin_df = bin_multilabel(compartment_df, 5)
    # bin_df = bin_df + 1
    # bin_df = bin_df.drop(columns='0+1_bin')
    # import seaborn as sns
    # sns.histplot(bin_df, multiple="dodge")
    # plt.show()

    bin_df = compartment_df.idxmax(axis=1).astype(int)
    bin_df.name = 'bin'
    bin_df = pd.DataFrame(bin_df)
    # bin_score = bin_df.sum(axis=1)
    # bin_df['entropy'] = entropy(bin_df.T)
    # bin_df['entropy'] = bin_df['entropy'].fillna(-1)
    # bin_df['score'] = bin_score
    bin_df = pd.concat([bin_df, compartment_df], axis=1)
    bin_df = bin_df.reset_index(names='barcode')
    # import seaborn as sns
    # sns.histplot(bin_df['score'], multiple="dodge")
    # plt.show()
    # plt.scatter(bin_df['entropy'], bin_df['score'])
    # plt.show()
    # sns.histplot(bin_df['entropy'], multiple="dodge")
    # plt.show()

    # stratified augmentation
    max_augs = 4
    score_frequency = bin_df['bin'].value_counts()
    max_frequency = score_frequency.max()
    augmentation_factors = max_frequency / score_frequency
    augmentation_factors = augmentation_factors.round().astype(int)
    # subtract to get the number of examples to augment
    augmentation_factors = augmentation_factors.astype(int) - 1
    # cap the augment numbers with max_augs
    augmentation_factors[augmentation_factors > max_augs] = max_augs
    augmentation_dict = dict(augmentation_factors)
    print(score_frequency)
    print(augmentation_factors)

    # import seaborn as sns
    # sns.histplot(bin_df, multiple="dodge")
    # plt.show()

    augmentations = [
        lambda patch: np.rot90(patch, axes=(1, 2)),  # Rotate 90 degrees
        lambda patch: np.rot90(patch, k=2, axes=(1, 2)),  # Rotate 180 degrees
        lambda patch: np.rot90(patch, k=3, axes=(1, 2)),  # Rotate 270 degrees
        lambda patch: np.fliplr(patch),  # Horizontal flip
        lambda patch: np.flipud(patch),  # Vertical flip
        lambda patch: np.fliplr(np.rot90(patch, axes=(1, 2))),  # Horizontal flip + Rotate 90
        lambda patch: np.fliplr(np.rot90(patch, k=2, axes=(1, 2))),  # Horizontal flip + Rotate 180
        lambda patch: np.fliplr(np.rot90(patch, k=3, axes=(1, 2))),  # Horizontal flip + Rotate 270
    ]

    # slice = tomography_volume[11]
    # plt.imshow(slice)
    # plt.scatter(spatial_coords_3d[:, 0], spatial_coords_3d[:, 1], s=2, c='tab:orange')
    # plt.show()

    # todo: do some function from this it is ugly
    # save patches in chunks
    max_size = 150
    n_sections = math.ceil(len(spatial_coords_3d) / max_size)
    splits = np.array_split(spatial_coords_3d, n_sections)  # creates near equal stuff
    y_splits = np.array_split(bin_df, n_sections)  # split label df

    for split_idx, data_split in tqdm(enumerate(splits), desc='Saving patch array splits to zarr...',
                                      total=len(splits)):
        # extract patches from volume
        if split_counter == 0:
            zarr_overwrite = True
        else:
            zarr_overwrite = False
        split_labels = y_splits[split_idx]
        patches = get_patches3d_dask(data_split, tomography_volume, radius=width_height_px, z_radius=n_slices)

        # concat augmented patches to the end of the split along with the labels
        augmented_patches = []
        augmented_labels = {}
        i = 0
        for p, (idx, row) in enumerate(split_labels.iterrows()):
            # check if we need to augment
            if augmentation_dict[row['bin']] != 0:
                original_patch = patches[p]
                # sampled_augmentations = random.sample(augmentations, int(row['score']))
                sampled_augmentations = random.sample(augmentations, int(augmentation_dict[row['bin']]))
                for f in sampled_augmentations:
                    augmented_patch = f(original_patch)
                    augmented_patches.append(augmented_patch)
                    augmented_labels[i] = row
                    i += 1

        augmented_patches = np.array(augmented_patches)
        augmented_labels = pd.DataFrame.from_dict(augmented_labels, orient='index')

        # combine patches with the augmented ones
        patches = np.expand_dims(patches, axis=-1)
        if augmented_patches.size > 0:
            augmented_patches = np.expand_dims(augmented_patches, axis=-1)
            patches = np.vstack([patches, augmented_patches])

        # combine labels
        split_labels['augment'] = False
        augmented_labels['augment'] = True
        split_labels = pd.concat([split_labels, augmented_labels])
        split_labels = split_labels[list(compartment_df.columns) + ['barcode', 'augment', 'bin']]
        split_labels = split_labels.reset_index(drop=True)
        split_labels['sample'] = x
        save_patches_to_zarr(patches, OUTPUT_PATCH, overwrite=zarr_overwrite)  # function is for 5D arrays!
        split_labels.to_csv(OUTPUT_PATCH.split('.zarr')[0] + f'_split_{split_counter}_{x}.csv')  # save labels
        split_counter += 1

# save patch shape and get rid of them to save memory
# patches_shape = patches.shape  - safe to delete

del patches
gc.collect()  # not sure if we need collection, looks good without
# todo: function end

#%%
EXPERIMENT = 'single_sample_v3_75epoch_64-64-21'
patch_shape = (n_slices*2+1, width_height_px*2, width_height_px*2, 1)

# read back the labels
import re
labels_df = pd.DataFrame()
csv_list = glob(OUTPUT_PATCH.split('.zarr')[0] + '*_split_*.csv')

def natural_sort_key(file_path):
    match = re.search(r"_split_(\d+)", file_path)
    return int(match.group(1)) if match else float('inf')  # Fallback to inf if no match

sorted_paths = sorted(csv_list, key=natural_sort_key)
for scsv in sorted_paths:
    print(f'Reading {scsv}')
    labels_df = pd.concat([labels_df, pd.read_csv(scsv, index_col=0)], ignore_index=True)

# do train test split per sample
selected_labels = [str(x) for x in range(8)]
y = labels_df[selected_labels].copy()
# Split data indexes
X_train, X_val, X_test = [], [], []
y_train, y_val, y_test  = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for s in ['L22_109_26']:
    labels_df_sample = labels_df[labels_df['sample'] == s]
    indexes = list(labels_df_sample.index)
    y_sample = labels_df_sample[selected_labels + ['bin']]

    # split barcodes by aggregating patches for each
    barcode_groups = pd.DataFrame(labels_df_sample.groupby('barcode').apply(lambda x: x.index.tolist()))
    bin_dict = {k:v for k, v in zip(labels_df_sample['barcode'], labels_df_sample['bin'])}
    barcode_groups['bin'] = [bin_dict[x] for x in barcode_groups.index]

    barcode_groups_train, barcode_groups_temp = train_test_split(barcode_groups, test_size=0.2, random_state=42,
                                                                 stratify=barcode_groups['bin'])
    barcode_groups_val, barcode_groups_test = train_test_split(barcode_groups_temp, test_size=0.3, random_state=42,
                                                               stratify=barcode_groups_temp['bin'])

    # flatten
    X_train.extend([index for group in barcode_groups_train[0] for index in group])
    X_val.extend([index for group in barcode_groups_val[0] for index in group])
    X_test.extend([index for group in barcode_groups_test[0] for index in group])

    # filter out the introduced augmentations from the training and validation set
    labels_df_sample_noaug = labels_df_sample[labels_df_sample['augment'] == False]
    X_val = [x for x in X_val if x in labels_df_sample_noaug.index]
    X_test = [x for x in X_test if x in labels_df_sample_noaug.index]



#     indexes = list(labels_df.index)
#     X_train_b, X_temp, y_train_b, y_temp = train_test_split(indexes, y, test_size=0.2, random_state=42,
#                                                             stratify=y['bin'])
#     X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42,
#                                                             stratify=y_temp['bin'])
#     print(len(X_train_b))
#     X_train.extend(X_train_b)
#     X_val.extend(X_val_b)
#     X_test.extend(X_test_b)
#     y_train = pd.concat([y_train, y_train_b], ignore_index=True)
#     y_test = pd.concat([y_test, y_test_b], ignore_index=True)
#     y_val = pd.concat([y_val, y_val_b], ignore_index=True)
#
# # this is only needed for the old approach to get rid of the bin column used for stratified splitting
# y = labels_df[selected_labels]

# Create datasets
batch_size = 16
buffer_size = 800
num_labels = len(selected_labels)

# this is only for eval
# get rid of the augmentation
labels_df_sample = labels_df[labels_df['sample'] == 'L22_109_26']
labels_df_sample = labels_df_sample[labels_df_sample['augment'] == False]
val_labels_df = labels_df_sample.drop_duplicates().copy()

val_labels_df.to_csv(SAVE_DIR + f'{EXPERIMENT}_labels.csv')

# not sure if this line is needed here, let's see
adata = sc.read_h5ad(INPUT_FOLDER + 'L22_109_26/adata_spatial_transformed_4.h5ad')
adata = adata[adata.obs.index.isin(list(val_labels_df['barcode']))]
val_labels_df['barcode'] = pd.Categorical(val_labels_df['barcode'], categories=adata.obs.index, ordered=True)
val_labels_df = val_labels_df.sort_values('barcode')
val_indexes = list(val_labels_df.index)
val_y = val_labels_df[selected_labels].values

whole_dataset = load_patch_and_labels_no_shuffle(OUTPUT_PATCH, val_indexes, y.values, 1,
                                      patch_shape[0], patch_shape[1], patch_shape[2], num_labels=num_labels)

# train-test-val datasets
train_dataset = load_patch_and_labels(OUTPUT_PATCH, X_train, y.values, batch_size,
                                      patch_shape[0], patch_shape[1], patch_shape[2],
                                      num_labels=num_labels, buffer_size=buffer_size)

validation_dataset = load_patch_and_labels(OUTPUT_PATCH, X_val, y.values, batch_size,
                                           patch_shape[0], patch_shape[1], patch_shape[2],
                                           num_labels=num_labels, buffer_size=buffer_size)

test_dataset = load_patch_and_labels(OUTPUT_PATCH, X_test, y.values, batch_size,
                                     patch_shape[0], patch_shape[1], patch_shape[2],
                                     num_labels=num_labels, buffer_size=buffer_size)

# debug
# for batch in train_dataset.take(1):
#     print(f"{batch[0].shape}, {batch[1].shape}")
#     plot_first_slices_with_titles(batch)
# for patch, label in train_dataset.take(1):
#     print(f"Patch batch shape: {patch.shape}, Label batch shape: {label.shape}")
# for batch in validation_dataset.take(1):
#     print(f"{batch[0].shape}, {batch[1].shape}")
# for batch in whole_dataset.take(1):
#     print(f"{batch[0].shape}, {batch[1].shape}")

#%%

# Create and train the model
model = create_3d_model_multi_output(input_shape=(patch_shape[0], patch_shape[1],
                                                  patch_shape[2], patch_shape[3]),
                                     n_output=8)
model.summary()

# keras doesn't know how many batches are there in the generator so we have to add it manually
steps_per_epoch = math.floor(len(X_train) / batch_size)
validation_steps = math.floor(len(X_val) / batch_size)
test_steps = math.floor(len(X_test) / batch_size)
whole_steps = math.floor(len(val_indexes) / 1)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=75, verbose=1,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    callbacks=[early_stopping])

predictions = model.predict(whole_dataset, steps=whole_steps)

plot_training_history(history, SAVE_DIR + f'{EXPERIMENT}_training_history.png')
plot_spatial(adata, SAVE_DIR + f'{EXPERIMENT}_spatial_comaprison.png',
             val_y, predictions)

with open(SAVE_DIR + f'{EXPERIMENT}_history.json', 'w') as f:
    json.dump(history.history, f)
model.save(SAVE_DIR + f'{EXPERIMENT}_multiclass.keras')

# get total r squared
total_dict = {}
for i in range(val_y.shape[1]):
    r_squared = r2_score(val_y[:, i], predictions[:, i])
    total_dict[i] = r_squared

# get other r squared stuff
r_squared_dict = {}
r_squared_dict['Total'] = total_dict

for labels, name in zip([X_train, X_val, X_test], ['Train', 'Val', 'Test']):
    r_batch = 32
    print(len(labels))

    # y = labels_df[selected_labels]
    # labels_df_sample = labels_df[labels_df['sample'] == 'L22_109_26']
    # val_labels_df = labels_df_sample.drop_duplicates().copy()

    temp_dataset = load_patch_and_labels_no_shuffle(OUTPUT_PATCH, labels, y.values, r_batch,
                                                    patch_shape[0], patch_shape[1], patch_shape[2],
                                                    num_labels=num_labels)
    temp_y = y.values[labels]
    predictions = model.predict(temp_dataset, steps=math.ceil(len(temp_y) / r_batch))

    # save predictions with the sample labels
    sample_vec = labels_df.loc[labels]['sample']
    perdictions_df = pd.DataFrame(predictions, index=sample_vec.index)
    perdictions_df['sample'] = sample_vec
    perdictions_df['barcode'] = labels_df.loc[labels]['barcode']
    perdictions_df.to_csv(SAVE_DIR + f'{EXPERIMENT}_predictions_{name}.csv')

    temp_dict = {}
    for i in range(temp_y.shape[1]):
        r_squared = r2_score(temp_y[:, i], predictions[:, i])
        temp_dict[i] = r_squared
    r_squared_dict[name] = temp_dict

r_squared_df = pd.DataFrame(data=r_squared_dict)
r_squared_df.to_csv(SAVE_DIR + f'{EXPERIMENT}_r_squared.csv')
