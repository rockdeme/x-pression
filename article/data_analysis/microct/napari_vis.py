import napari
import dask.array as da
import tifffile as tiff
import numpy as np

# save_file = '/mnt/f/3d_tomography/cnn_training/' + "stride32.zarr"
save_file = 'F:\\3d_tomography\\cnn_training\\' + "stride16_downscaled.tiff"

colormaps = ['#ffffff', '#dbc257', '#5770db', '#db5f57', '#db57b2', '#91db57', '#57d3db', '#57db80', '#a157db']

scale_factor = 32

scale = (21.0 / scale_factor, 1.0, 1.0)
num_channels = 9
with tiff.TiffFile(save_file) as tif:
    num_pages = len(tif.pages)
    height, width = tif.pages[0].shape
    Z = num_pages // num_channels  # Number of Z slices

    # Read and stack slices into a (Z, C, H, W) array
    image_4d = np.stack([
        np.stack([tif.pages[i * num_channels + c].asarray() for c in range(num_channels)], axis=0)
        for i in range(Z)
    ], axis=0)

napari.view_image(image_4d, channel_axis=1, colormap=colormaps, scale=scale)

if __name__ == '__main__':
    napari.run()
