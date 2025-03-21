import cv2
import zarr
import numpy as np
from tqdm import tqdm
import chrysalis as ch
import tifffile as tiff
import dask.array as da
import matplotlib.pyplot as plt


def write_tiff_to_zarr(input_file, save_file, chunks=(1, 1, 1024, 1024), num_channels=9, downsample_factor=None,
                       x_clip=None, y_clip=None, z_clip=None, convert_to_int=False):
    with tiff.TiffFile(input_file) as tif:
        height, width = tif.pages[0].shape
        depth = len(tif.pages)
        Z = depth // num_channels
        if x_clip:
            width = x_clip[1] - x_clip[0]
        if downsample_factor:
            height, width = height // downsample_factor, width // downsample_factor

        zarr_array = zarr.create(shape=(Z, num_channels, height, width),
                                 chunks=chunks,  # (Z, C, W, H)
                                 dtype=float,
                                 store=save_file,
                                 overwrite=True)

        for i in tqdm(range(Z), 'Writing slices to zarr...'):
            slice_factor = i * num_channels
            slices = np.stack([page.asarray() for page in
                               tif.pages[slice_factor:slice_factor + num_channels]], axis=0)
            slices = slices.astype(np.float32)
            if x_clip:
                slices = slices[:, :, x_clip[0]:x_clip[1]]
            if downsample_factor:
                if not x_clip:
                    slices = downsample_img(slices.compute(), downsample_factor)
                else:
                    slices = downsample_img(slices, downsample_factor)
            if convert_to_int:
                slices = (slices * 255).astype(np.uint8)
            zarr_array[i] = slices


def write_tiff_to_tiff(input_file, output_file, num_channels=9, downsample_factor=None,
                       x_clip=None, y_clip=None, z_clip=None, convert_to_int=False, gaussian=None):
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

                if downsample_factor:
                    slices = slices[:, ::downsample_factor, ::downsample_factor]

                background = slices[1:].sum(axis=0) == 0   # (Z, C, W, H)

                if gaussian:
                    for idx, s in zip(range(1, len(slices)), slices[1:]):
                        print(idx, str(np.sum(s)))
                        slices[idx] = cv2.GaussianBlur(s, (gaussian, gaussian), 0)

                slices[:, background] = 0.0

                if convert_to_int:
                    slices = (slices * 255).astype(np.uint8)

                for slice_img in slices:
                    tif_writer.write(slice_img, photometric='minisblack')


def downsample_img(array, factor):
    downsampled = []
    for s in array:
        downsampled_im = cv2.resize(s, (s.shape[1] // factor, s.shape[0] // factor), interpolation=cv2.INTER_AREA)
        downsampled.append(downsampled_im)
    downsampled = np.stack(downsampled, axis=0)
    return downsampled

def mip_colors(colors_1, colors_2):
    colors_1 = np.array(colors_1)
    colors_2 = np.array(colors_2)
    mip_color = np.maximum(colors_1, colors_2)
    return mip_color

def map_colors(array, seed=42):
    shapes = array.shape
    assert len(shapes) == 3, 'Specify a 3D numpy array (C, X, Y)'
    hexcodes = ch.plots.get_hexcodes(None, shapes[0], seed, 1)
    cmaps = []
    for d in range(shapes[0]):
        pc_cmap = ch.utils.black_to_color(hexcodes[d])
        pc_rgb = ch.utils.get_rgb_from_colormap(pc_cmap, vmin=0, vmax=1, value=array[d].flatten())
        cmaps.append(pc_rgb)

    cblend = mip_colors(cmaps[0], cmaps[1])
    if len(cmaps) > 2:
        i = 2
        for cmap in cmaps[2:]:
            cblend = mip_colors(cblend, cmap)
            i += 1
    cblend = np.reshape(cblend, (shapes[1], shapes[2], 3))
    return cblend

def create_composite(slice, seed=42, white=True):
    rgb = map_colors(slice[1:], seed)
    if white:
        rgb[rgb.sum(axis=2) == 0] = 1
    return rgb


def adjust_brightness(img, percent):
    factor = percent / 100.0
    img = img * 255
    img = img.astype(np.uint8)
    img = img + (factor * 255)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def generate_overlay(zarr_image, slice_id, plane='axial', Z_step=21, downscale_factor=2, uint=False):

    if plane == 'axial':
        slice = zarr_image[slice_id]
    elif plane == 'sagittal':
        sagittal_slice = zarr_image[:, :, slice_id, :]
        sagittal_slice = sagittal_slice.repeat(Z_step, axis=0)
        slice = np.swapaxes(sagittal_slice, 1, 0)  # swap to (C, W, H)
    elif plane == 'coronal':
        coronal_slice = zarr_image[:, :, :, slice_id]
        coronal_slice = coronal_slice.repeat(Z_step, axis=0)
        slice = np.swapaxes(coronal_slice, 1, 0)  # swap to (C, W, H)
    else:
        raise Exception('Not valid but I am too lazy to type the options so just check the function.')

    if type(slice) == np.ndarray:
        downsampled_slice = downsample_img(slice, downscale_factor)
    else:
        downsampled_slice = downsample_img(slice.compute(), downscale_factor)
    if uint:
        downsampled_slice = downsampled_slice / 255
    channels = downsampled_slice[1:, :, :]
    tomogram = downsampled_slice[0]

    tissue = channels.sum(axis=0)
    tomogram[tissue == 0] = 0

    # min-max scale
    nonzero_pixels = tomogram[tomogram > 0]
    if len(nonzero_pixels) > 0:
        min_val, max_val = np.percentile(nonzero_pixels, (1, 99))
        tomogram = np.clip((tomogram - min_val) / (max_val - min_val), 0, 1)

    tomogram_rgb = np.stack([tomogram] * 3, axis=-1)  # Shape: (W, H, 3)

    composite = create_composite(downsampled_slice)

    # add the colors to the tomogram
    colored_overlay = tomogram_rgb * composite * 3  # multiply the composite to make it brighter
    colored_overlay[colored_overlay > 1.0] = 1.0

    return colored_overlay, tomogram_rgb

#%%
# input_file = '/mnt/d/data/covid/tomography/prediction/combined_slices_depth11_stride32_0and1.tiff'
input_file = '/mnt/f/3d_tomography/cnn_training/combined_slices_depth11_stride16_0and1.tiff'
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16.zarr"
num_channels = 9
Z_step = 21
voxel_size = 1.6  # only needed for plotting

x_clip = [1416, 10656]
write_tiff_to_zarr(input_file, save_file, chunks=(1, 9, 512, 512), num_channels=9, x_clip=x_clip)

save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_downscaled.zarr"
write_tiff_to_zarr(input_file, save_file, chunks=(1, 9, 512, 512), num_channels=9, downsample_factor=4, x_clip=x_clip,
                   convert_to_int=False)

# for imaris
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_gauss65.tiff"
write_tiff_to_tiff(input_file, save_file, num_channels=9, x_clip=x_clip, gaussian=65,
                   convert_to_int=True)

# downscaled tiff
save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16_downscaled.tiff"
write_tiff_to_tiff(input_file, save_file, num_channels=9, downsample_factor=4, x_clip=x_clip,
                   convert_to_int=True)

# only ct - all slices
x_clip = [1416, 10656]
input_file = '/mnt/d/data/covid/tomography/' + "ct_extracted_masked.tiff"
save_file = '/mnt/d/data/covid/tomography/' + "segmented_all_slices.zarr"
write_tiff_to_zarr(input_file, save_file, chunks=(10, 1, 1024, 1024), num_channels=1, x_clip=x_clip)

# sample 1096
input_file = '/mnt/f/3d_tomography/cnn_training/sample1096/combined_slices_depth11_stride16_0and1.tiff'
save_file = '/mnt/f/3d_tomography/cnn_training/' + "1096.zarr"

input_file = ('/mnt/f/3d_tomography/cnn_training/combined_slices_depth11_stride16_0and1_single/'
              'combined_slices_depth11_stride16_0and1.tiff')
save_file = '/mnt/f/3d_tomography/cnn_training/' + "1096_single_sample.zarr"

with tiff.TiffFile(input_file) as tif:
    height, width = tif.pages[0].shape
    depth = len(tif.pages)
    slice_img = tif.pages[depth // 4  * 3].asarray()
plt.imshow(slice_img)
plt.show()

x_clip = [1000, 7666]
write_tiff_to_zarr(input_file, save_file, chunks=(1, 9, 512, 512), num_channels=9, x_clip=x_clip)

# sample 1210
input_file = '/mnt/f/3d_tomography/cnn_training/sample1210/sample1210.tiff'
save_file = '/mnt/f/3d_tomography/cnn_training/' + "1210.zarr"

with tiff.TiffFile(input_file) as tif:
    height, width = tif.pages[0].shape
    depth = len(tif.pages)
    slice_img = tif.pages[depth // 4  * 1].asarray()
plt.imshow(slice_img)
plt.show()

x_clip = [1000, 9750]
write_tiff_to_zarr(input_file, save_file, chunks=(1, 9, 512, 512), num_channels=9, x_clip=x_clip)

# sample L2210914
input_file = '/mnt/f/3d_tomography/cnn_training/L2210914/L2210914.tiff'
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914.zarr"

with tiff.TiffFile(input_file) as tif:
    height, width = tif.pages[0].shape
    depth = len(tif.pages)
    slice_img = tif.pages[depth // 8  * 1].asarray()
plt.imshow(slice_img)
plt.show()

x_clip = [250, 9000]
write_tiff_to_zarr(input_file, save_file, chunks=(1, 9, 512, 512), num_channels=9, x_clip=x_clip)

# for imaris
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_gauss33.tiff"
write_tiff_to_tiff(input_file, save_file, num_channels=9, x_clip=x_clip, gaussian=33,
                   convert_to_int=True)

# only ct - all slices
x_clip = [250, 9000]
input_file = '/mnt/f/3d_tomography/cnn_training/L2210914/' + "ct_extracted_masked.tiff"
save_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_segmented_all_slices.zarr"
write_tiff_to_zarr(input_file, save_file, chunks=(10, 1, 1024, 1024), num_channels=1, x_clip=x_clip)

#%%
zarr_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16.zarr"
save_folder = 'data/figures/926/'
# reload from Zarr
zarr_image = da.from_zarr(zarr_file)

axial_images = []
for slice_id in tqdm([2, 4, 6, 10]):
    axial_slice = zarr_image[slice_id]
    axial_slice = downsample_img(axial_slice.compute(), 2)
    axial_slice = create_composite(axial_slice)
    axial_slice = axial_slice * 1.25  # multiply the composite to make it brighter
    axial_slice[axial_slice > 1.0] = 1.0
    plt.imsave(save_folder + f'axial_{slice_id}_mip.png', axial_slice)

selected_axial = np.arange(10, 60, 10)
selected_coronal = np.arange(2000, zarr_image.shape[3], 1250)
selected_sagittal = np.arange(750, zarr_image.shape[2], 750)

axial_images = []
for slice_id in tqdm(selected_axial):
    axial_slice = zarr_image[slice_id]
    axial_slice = downsample_img(axial_slice.compute(), 2)
    axial_slice = create_composite(axial_slice)
    axial_images.append(axial_slice)

fig, ax = plt.subplots(5, 1, figsize=(3.5, 10))
for i, s in enumerate(selected_axial):
    ax[i].imshow(adjust_brightness(axial_images[i], 10))
    ax[i].axis('off')
    ax[i].set_title(f'Z: {s * voxel_size * Z_step} μm')
plt.tight_layout()
plt.show()

coronal_images = []
for slice_id in tqdm(selected_coronal):
    coronal_slice = zarr_image[:, :, :, slice_id]
    coronal_slice = coronal_slice.repeat(Z_step, axis=0)
    coronal_slice = np.swapaxes(coronal_slice, 1, 0)  # swap to (C, W, H)
    coronal_slice = downsample_img(coronal_slice.compute(), 2)
    coronal_slice = create_composite(coronal_slice)
    coronal_images.append(coronal_slice)

fig, ax = plt.subplots(6, 1, figsize=(1.5, 10))
for i, s in enumerate(selected_coronal):
    ax[i].imshow(np.rot90(adjust_brightness(coronal_images[i], 10), 3))
    ax[i].axis('off')
    ax[i].set_title(f'X: {s * voxel_size / 1000:.1f} mm')
plt.tight_layout()
plt.show()

sagittal_images = []
for slice_id in tqdm(selected_sagittal):
    sagittal_slice = zarr_image[:, :, slice_id, :]
    sagittal_slice = sagittal_slice.repeat(Z_step, axis=0)
    sagittal_slice = np.swapaxes(sagittal_slice, 1, 0)  # swap to (C, W, H)
    sagittal_slice = downsample_img(sagittal_slice.compute(), 2)
    sagittal_slice = create_composite(sagittal_slice)
    sagittal_images.append(sagittal_slice)

fig, ax = plt.subplots(6, 1, figsize=(3.5, 5))
for i, s in enumerate(selected_sagittal):
    ax[i].imshow(adjust_brightness(sagittal_images[i], 10))
    ax[i].axis('off')
    ax[i].set_title(f'Y: {s * voxel_size / 1000:.1f} mm')
plt.tight_layout()
plt.show()

plt.imshow(axial_images[2])
plt.show()
plt.imsave('/mnt/f/3d_tomography/cnn_training/asd.png', axial_images[2])
plt.show()

#%%
# overlay plots

zarr_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16.zarr"
save_folder = 'data/figures/926/'
# reload from Zarr
zarr_image = da.from_zarr(zarr_file)

selected_axial = np.arange(10, 60, 10)
selected_coronal = np.arange(2000, zarr_image.shape[3], 1250)
selected_sagittal = np.arange(750, zarr_image.shape[2], 750)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial')
    plt.imsave(save_folder + f'axial_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram.png', tomogram)

# small ones for the series
for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', downscale_factor=6)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_small.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram_small.png', tomogram)

for slice_id in tqdm(selected_coronal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='coronal')
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_sagittal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='sagittal')
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)

# rectangle
file_path = '/mnt/f/3d_tomography/cnn_training/imaris/long_rectangle_hyperstack.tif'
save_folder = 'data/figures/926_long/'
with tiff.TiffFile(file_path) as tif:
    slices = tif.asarray()
    print(slices.shape)

plt.hist(slices[:, 0, :, :].std(axis=(-2, -1)), bins=100)
plt.show()

selected_axial = np.arange(10, 1230, 300)
selected_coronal = np.arange(0, slices.shape[3], 100)
selected_sagittal = np.arange(0, slices.shape[2], 100)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='axial')
    plt.imsave(save_folder + f'axial_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_coronal):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='coronal', Z_step=1)
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_sagittal):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='sagittal', Z_step=1)
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)

# min max
for slice_id in tqdm([0, slices.shape[0]-1]):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='axial')
    plt.imsave(save_folder + f'axial_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm([0, slices.shape[3]-1]):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='coronal', Z_step=1)
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm([0, slices.shape[2]-1]):
    overlay, tomogram = generate_overlay(slices, slice_id, plane='sagittal', Z_step=1)
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)

# 1096
file_path = '/mnt/f/3d_tomography/cnn_training/' + '1096.zarr'
save_folder = 'data/figures/1096/'
zarr_image = da.from_zarr(file_path)

selected_axial = np.arange(10, 60, 10)
selected_coronal = np.arange(2000, zarr_image.shape[3], 1250)
selected_sagittal = np.arange(750, zarr_image.shape[2], 750)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial')
    plt.imsave(save_folder + f'axial_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_coronal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='coronal', Z_step=21)
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_sagittal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='sagittal', Z_step=21)
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)

# 1096
file_path = '/mnt/f/3d_tomography/cnn_training/' + '1096_single_sample.zarr'
save_folder = 'data/figures/1096/'
zarr_image = da.from_zarr(file_path)

selected_axial = np.arange(10, 60, 10)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=True)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample.png', overlay)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=True, downscale_factor=6)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample_small.png', overlay)

# 1210
file_path = '/mnt/f/3d_tomography/cnn_training/' + '1210.zarr'
save_folder = 'data/figures/1210/'
zarr_image = da.from_zarr(file_path)

selected_axial = np.arange(10, 60, 10)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=False)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample.png', overlay)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=False, downscale_factor=6)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample_small.png', overlay)

# sample L2210914
file_path = '/mnt/f/3d_tomography/cnn_training/' + "L2210914.zarr"
save_folder = 'data/figures/L2210914/'
zarr_image = da.from_zarr(file_path)

selected_axial = np.arange(10, 60, 10)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=False)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_axial):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='axial', uint=False, downscale_factor=6)
    plt.imsave(save_folder + f'axial_{slice_id}_overlay_single_sample_small.png', overlay)
    plt.imsave(save_folder + f'axial_{slice_id}_tomogram_small.png', tomogram)

#%%
# downsample stuff

save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16.zarr"
output_file =  '/mnt/f/3d_tomography/cnn_training/' + "cropped_rectangle_2.tiff"
# reload from Zarr
zarr_image = da.from_zarr(save_file)
selected_axial = range(zarr_image.shape[0])

Z_step = 21
y_crop_s, y_crop_e = 3124, 3624
x_crop_s, x_crop_e = 2715, 3215

axial_images = []
for slice_id in tqdm(selected_axial):
    axial_slice = zarr_image[slice_id, :, y_crop_s:y_crop_e, x_crop_s:x_crop_e]
    # axial_slice = axial_slice.repeat(Z_step, axis=0)
    axial_images.append(axial_slice)

axial_images = np.stack(axial_images)

with tiff.TiffWriter(output_file, bigtiff=True) as tif_writer:
    for slice_img in axial_images:
        tif_writer.write(slice_img, photometric='minisblack')


axial_slice = zarr_image[30, 0, :, :]
plt.imshow(axial_slice)
plt.show()

axial_slice = zarr_image[30, 0, y_crop_s:y_crop_e, x_crop_s:x_crop_e]
plt.imshow(axial_slice)
plt.show()


# debug
file_path = '/mnt/f/3d_tomography/cnn_training/' + '1096_single_sample.zarr'
save_folder = 'data/figures/1096/'
zarr_image = da.from_zarr(file_path)

axial_images = []
for slice_id in tqdm([2, 4, 6, 10]):
    axial_slice = zarr_image[slice_id]
    axial_slice = downsample_img(axial_slice.compute(), 2)
    axial_slice = create_composite(axial_slice)
    axial_slice = axial_slice * 1.25  # multiply the composite to make it brighter
    axial_slice[axial_slice > 1.0] = 1.0
    plt.imsave(save_folder + f'axial_{slice_id}_mip_single_sample.png', axial_slice)