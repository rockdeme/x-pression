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


def generate_overlay(zarr_image,slice_id, plane='axial', Z_step=21, downscale_factor=2, uint=False, tomogram_zarr=None):

    if tomogram_zarr is not None:
        if plane == 'axial':
            tomogram = tomogram_zarr[slice_id * Z_step]
        elif plane == 'sagittal':
            sagittal_slice = tomogram_zarr[:, :, slice_id, :]
            tomogram = np.swapaxes(sagittal_slice, 1, 0)  # swap to (C, W, H)
            tomogram = tomogram[:, :-Z_step, :]
        elif plane == 'coronal':
            coronal_slice = tomogram_zarr[:, :, :, slice_id]
            tomogram = np.swapaxes(coronal_slice, 1, 0)  # swap to (C, W, H)
            tomogram = tomogram[:, :-Z_step, :]

        else:
            raise Exception('Not valid but I am too lazy to type the options so just check the function.')

        if type(tomogram) == np.ndarray:
            tomogram = downsample_img(tomogram, downscale_factor)
        else:
            tomogram = downsample_img(tomogram.compute(), downscale_factor)
        if uint:
            tomogram = tomogram / 255
        tomogram = tomogram[0]

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

    if tomogram_zarr is None:
        tomogram = downsampled_slice[0]

    downsampled_slice = downsampled_slice[:, :tomogram.shape[0], :]
    channels = downsampled_slice[1:, :, :]

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
# overlay plots

tomogram_file = '/mnt/d/data/covid/tomography/' + "segmented_all_slices.zarr"
zarr_file = '/mnt/f/3d_tomography/cnn_training/' + "stride16.zarr"
save_folder = 'data/figures/926/'
# reload from Zarr
zarr_image = da.from_zarr(zarr_file)
tomogram_zarr = da.from_zarr(tomogram_file)

selected_axial = np.arange(10, 60, 10)
selected_coronal = np.arange(2000, zarr_image.shape[3], 1250)
selected_sagittal = np.arange(750, zarr_image.shape[2], 750)

for slice_id in tqdm(selected_coronal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='coronal', tomogram_zarr=tomogram_zarr)
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_sagittal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='sagittal', tomogram_zarr=tomogram_zarr)
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)


# L2210914
tomogram_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914_segmented_all_slices.zarr"
zarr_file = '/mnt/f/3d_tomography/cnn_training/' + "L2210914.zarr"
save_folder = 'data/figures/L2210914/'
# reload from Zarr
zarr_image = da.from_zarr(zarr_file)
tomogram_zarr = da.from_zarr(tomogram_file)

selected_axial = np.arange(10, 60, 10)
selected_coronal = np.arange(2000, zarr_image.shape[3], 1250)
selected_sagittal = np.arange(750, zarr_image.shape[2], 750)

for slice_id in tqdm(selected_coronal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='coronal', tomogram_zarr=tomogram_zarr)
    plt.imsave(save_folder + f'coronal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'coronal_{slice_id}_tomogram.png', tomogram)

for slice_id in tqdm(selected_sagittal):
    overlay, tomogram = generate_overlay(zarr_image, slice_id, plane='sagittal', tomogram_zarr=tomogram_zarr)
    plt.imsave(save_folder + f'sagittal_{slice_id}_overlay.png', overlay)
    plt.imsave(save_folder + f'sagittal_{slice_id}_tomogram.png', tomogram)
