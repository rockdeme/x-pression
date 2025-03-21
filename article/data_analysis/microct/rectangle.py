import cv2
import zarr
import numpy as np
from tqdm import tqdm
import chrysalis as ch
import tifffile as tiff
import dask.array as da
import matplotlib.pyplot as plt


file_path = '/mnt/f/3d_tomography/cnn_training/imaris/combined_slices_128x128_stride16-1.tif'
output_file = ''

with tiff.TiffFile(file_path) as tif:
    slices = tif.asarray()

slices = slices[:int(slices.shape[0] / 16) * 16]
slices = slices.reshape(slices.shape[0] / 16, 16, 9, 384, 384).mean(axis=1)
z_expanded = np.repeat(z_new, 16, axis=0)




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

        background = slices[1:].sum(axis=0) == 0  # (Z, C, W, H)

        if gaussian:
            for idx, s in zip(range(1, len(slices)), slices[1:]):
                print(idx, str(np.sum(s)))
                slices[idx] = cv2.GaussianBlur(s, (gaussian, gaussian), 0)

        slices[:, background] = 0.0

        if convert_to_int:
            slices = (slices * 255).astype(np.uint8)

        for slice_img in slices:
            tif_writer.write(slice_img, photometric='minisblack')
