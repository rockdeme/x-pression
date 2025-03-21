import keras.src.saving
import numpy as np
import tifffile as tiff
from glob import glob


slice_files = glob('/path/to/original/volume_tiff/*.tiff')
masks_files = glob('/path/to/binary_masks/*.npy')
model_name = 'unet_segmentation_v1_128_multisample_train1'
save_path = '/mnt/e/3d_tomography/Chang/results'

patches = []
masks = []
for slice_path, mask_path in zip(slice_files, masks_files):
    transformed_he_image_2 = tiff.imread(slice_path)
    # Load the binary mask
    filled_mask = np.load(mask_path)

    # Set parameters for patch extraction
    patch_size = (128, 128)

    # Calculate the area of the patch
    patch_area = patch_size[0] * patch_size[1]
    half_patch_area = patch_area / 2  # 50% of the patch area

    # Extract patches
    for i in range(0, transformed_he_image_2.shape[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, transformed_he_image_2.shape[1] - patch_size[1] + 1, patch_size[1]):
            patch = transformed_he_image_2[i:i + patch_size[0], j:j + patch_size[1]]
            patch_mask = filled_mask[i:i + patch_size[0], j:j + patch_size[1]]

            patches.append(patch)
            masks.append(patch_mask)

# Convert lists to numpy arrays
patches = np.array(patches)
masks = np.array(masks)

# Optionally save the patches to disk
# np.save('patches.npy', patches)
# np.save('masks.npy', masks)

# Print the shapes of the resulting arrays
print(f'Background patches shape: {patches.shape}')
print(f'Tissue patches shape: {masks.shape}')

patches = patches / 255.0
masks = np.expand_dims(masks, axis=-1)
patches = np.expand_dims(patches, axis=-1)

print(f'Background patches shape: {patches.shape}')
print(f'Tissue patches shape: {masks.shape}')

from keras import layers
from keras import models
def conv_block(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 8)
    s2, p2 = encoder_block(p1, 16)
    s3, p3 = encoder_block(p2, 32)
    s4, p4 = encoder_block(p3, 64)

    b1 = conv_block(p4, 128)

    d1 = decoder_block(b1, s4, 64)
    d2 = decoder_block(d1, s3, 32)
    d3 = decoder_block(d2, s2, 16)
    d4 = decoder_block(d3, s1, 8)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = models.Model(inputs, outputs, name="U-Net")
    return model

import tensorflow as tf

# Dice coefficient metric
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Dice loss for optimization
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# IOU metric function using TensorFlow's own operations
def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from keras import callbacks
from keras import optimizers
from keras import metrics
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(patches, masks, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

import matplotlib.pyplot as plt

def plot_first_slices_with_titles(batch):
    # images, labels = batch
    batch_size = batch.shape[0]  # 16 in your case
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))

    for i in range(batch_size):
        # Extract the first slice (shape: 64x64)
        first_slice = batch[i, :, :, 0]

        # Plot the first slice
        axes[i].imshow(first_slice, cmap='gray', vmin=0, vmax=1)

        # Add title (label)

        # Hide axis for cleaner look
        axes[i].axis('off')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()


def plot_training_history(history, save_path):
    """Save the training history plot to a file."""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['dice_coef']
    val_mae = history.history['val_dice_coef']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_loss, label='Training Loss (1-dice)', color='blue')
    axes[0].plot(val_loss, label='Validation Loss (1-dice)', color='orange')
    axes[0].set_title('Loss (1-dice) Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (1-dice)')
    axes[0].legend()

    axes[1].plot(train_mae, label='Training dice', color='blue')
    axes[1].plot(val_mae, label='Validation dice', color='orange')
    axes[1].set_title('Dice Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('dice')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

nn = 200
plot_first_slices_with_titles(X_val[nn:nn+16, :, :, :])
plot_first_slices_with_titles(y_val[nn:nn+16, :, :, :])


print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Ensure the input shape for the model matches (e.g., (128, 128, 1))
input_shape = X_train.shape[1:]  # (128, 128, 1)

batch_size = 4
lr = 1e-4
num_epochs = 200
# Build and compile the U-Net model
model = build_unet(input_shape)
metrics = [dice_coef, iou, metrics.Recall(), metrics.Precision()]

model.compile(loss=dice_loss, optimizer=optimizers.Adam(lr), metrics=metrics)
model.summary()
callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
    callbacks.TensorBoard(),
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False),
]

history = model.fit(
    X_train, y_train,
    epochs=num_epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True
)

model.save(f'{save_path}/{model_name}.keras')
plot_training_history(history, f'{save_path}/{model_name}_training_history.png')

#%%

patches = []
masks = []
for slice_path, mask_path in zip(slice_files, masks_files):
    # Define the path to the saved TIFF image
    transformed_he_image_2 = tiff.imread(slice_path)
    # Load the binary mask
    filled_mask = np.load(mask_path)

    # Set parameters for patch extraction
    patch_size = (128, 128)

    # Calculate the area of the patch
    patch_area = patch_size[0] * patch_size[1]
    half_patch_area = patch_area / 2  # 50% of the patch area

    # Extract patches
    for i in range(0, transformed_he_image_2.shape[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, transformed_he_image_2.shape[1] - patch_size[1] + 1, patch_size[1]):
            patch = transformed_he_image_2[i:i + patch_size[0], j:j + patch_size[1]]
            patch_mask = filled_mask[i:i + patch_size[0], j:j + patch_size[1]]

            patches.append(patch)
            masks.append(patch_mask)

# Convert lists to numpy arrays
patches = np.array(patches)
masks = np.array(masks)

# Optionally save the patches to disk
# np.save('patches.npy', patches)
# np.save('masks.npy', masks)

# Print the shapes of the resulting arrays
print(f'Background patches shape: {patches.shape}')
print(f'Tissue patches shape: {masks.shape}')

import keras

model = keras.src.saving.load_model(f'{save_path}/{model_name}.keras',
                                    custom_objects={'dice_coef': dice_coef,
                                                    'dice_loss': dice_loss,
                                                    'iou': iou})
#
# nn = 10
# plot_first_slices_with_titles(patches[nn:nn+16, :, :, :])
# plot_first_slices_with_titles(predictions[nn:nn+16, :, :, :])

def stitch_pca_predictions_into_image(pca_predictions, image_shape, patch_size, stride):
    stitched_image = np.zeros(image_shape, dtype=np.float32)

    patch_idx = 0
    half_stride = stride // 2

    # Iterate over the image using stride, not patch_size
    for i in range(0, image_shape[0] - patch_size + 1, stride):
        for j in range(0, image_shape[1] - patch_size + 1, stride):
            # if patch_idx < len(pca_predictions) and not np.isnan(pca_predictions[patch_idx]):
            intensity = pca_predictions[patch_idx]

            # Define the center of the patch
            center_i = i + patch_size // 2
            center_j = j + patch_size // 2

            # Calculate the patch's bounds based on half the stride
            i_start = max(0, center_i - half_stride)
            i_end = min(image_shape[0], center_i + half_stride)
            j_start = max(0, center_j - half_stride)
            j_end = min(image_shape[1], center_j + half_stride)

            # Add the intensity to the patch area
            stitched_image[i_start:i_end, j_start:j_end] += intensity

            patch_idx += 1

    return stitched_image

for idx in range(len(slice_files)):

    transformed_he_image_2 = tiff.imread(slice_files[idx])
    # Load the binary mask

    # Set parameters for patch extraction
    patch_size = (128, 128)

    # Calculate the area of the patch
    patch_area = patch_size[0] * patch_size[1]
    half_patch_area = patch_area / 2  # 50% of the patch area
    patches = []
    # Extract patches
    for i in range(0, transformed_he_image_2.shape[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, transformed_he_image_2.shape[1] - patch_size[1] + 1, patch_size[1]):
            patch = transformed_he_image_2[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)

    patches = np.array(patches)
    x = patches / 255.0
    x = np.expand_dims(x, axis=-1)
    predictions = model.predict(x)

    filled_mask = np.load(masks_files[idx])
    stitched_preds = stitch_pca_predictions_into_image(predictions[:, :, :, 0], transformed_he_image_2.shape, 128, 128)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].imshow(stitched_preds)
    axes[0].set_title("Stitched predictions")
    axes[1].imshow(filled_mask)
    axes[1].set_title("Filled mask")
    axes[2].imshow(transformed_he_image_2)
    axes[2].set_title("CT")
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_{idx}.png')
    plt.show()

#%%


import os
import sys
import argparse
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import tensorflow as tf  # Import TensorFlow
from keras import models
from keras import utils

save_path = '/mnt/e/3d_tomography/Chang/results'

@utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Dice loss for optimization
@utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

model = models.load_model(
    f'{save_path}/{model_name}.keras',
    custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss, 'iou': iou})

patch_size= 128
patches =[]
transformed_he_image_2 = tiff.imread('data/selected_full_size_slice2.tiff')
for i in range(0, transformed_he_image_2.shape[0] - patch_size + 1, patch_size):
    for j in range(0, transformed_he_image_2.shape[1] - patch_size + 1, patch_size):
        patch = transformed_he_image_2[i:i + patch_size, j:j + patch_size]
        patches.append(patch)

# Convert lists to numpy arrays
patches = np.array(patches)
predictions = model.predict(patches)

import matplotlib.pyplot as plt
def stitch_pca_predictions_into_image(pca_predictions, image_shape, patch_size, stride):
    stitched_image = np.zeros(image_shape, dtype=np.float32)

    patch_idx = 0
    half_stride = stride // 2

    # Iterate over the image using stride, not patch_size
    for i in range(0, image_shape[0] - patch_size + 1, stride):
        for j in range(0, image_shape[1] - patch_size + 1, stride):
            # if patch_idx < len(pca_predictions) and not np.isnan(pca_predictions[patch_idx]):
            intensity = pca_predictions[patch_idx]

            # Define the center of the patch
            center_i = i + patch_size // 2
            center_j = j + patch_size // 2

            # Calculate the patch's bounds based on half the stride
            i_start = max(0, center_i - half_stride)
            i_end = min(image_shape[0], center_i + half_stride)
            j_start = max(0, center_j - half_stride)
            j_end = min(image_shape[1], center_j + half_stride)

            # Add the intensity to the patch area
            stitched_image[i_start:i_end, j_start:j_end] += intensity

            patch_idx += 1

    return stitched_image

stitched_preds = stitch_pca_predictions_into_image(predictions[:, :, :, 0], transformed_he_image_2.shape, patch_size, patch_size)

plt.imshow(stitched_preds)
plt.show()