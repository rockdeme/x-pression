import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
import chrysalis as ch
from tqdm import tqdm


# input_file = '/mnt/d/data/covid/tomography/prediction/combined_slices_depth11_stride32_0and1.tiff'
input_file = '/mnt/f/3d_tomography/cnn_training/combined_slices_depth11_stride32_0and1.tiff'

slice_id = 40

slice_id *= 9  # number of compartments
with tiff.TiffFile(input_file) as tif:
    slices = np.stack([page.asarray() for page in tif.pages[slice_id+0:slice_id+9]], axis=0)

slices = slices.astype(np.float32)

downsampled = []
for s in slices:
    downsampled_im = cv2.resize(s, (s.shape[1] // 2, s.shape[0] // 2), interpolation=cv2.INTER_AREA)
    downsampled.append(downsampled_im)
downsampled = np.stack(downsampled, axis=0)

plt.imshow(downsampled[0])
plt.show()


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

rgb = map_colors(downsampled[1:], 42)
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
rgb[rgb.sum(axis=2) == 0] = 1
ax.imshow(rgb)
plt.show()

# do a bunch
num_channels = 9
with tiff.TiffFile(input_file) as tif:
    num_slices = len(tif.pages)
num_slices /= num_channels

selected_slices = np.arange(10, 60, 10)

images = []
for slice_id in tqdm(selected_slices):
    slice_id *= num_channels  # number of compartments
    with tiff.TiffFile(input_file) as tif:
        slices = np.stack([page.asarray() for page in tif.pages[slice_id+0:slice_id+9]], axis=0)
    slices = slices.astype(np.float32)

    downsampled = []
    for s in slices:
        downsampled_im = cv2.resize(s, (s.shape[1] // 2, s.shape[0] // 2), interpolation=cv2.INTER_AREA)
        downsampled.append(downsampled_im)
    downsampled = np.stack(downsampled, axis=0)

    rgb = map_colors(downsampled[1:], 42)
    rgb[rgb.sum(axis=2) == 0] = 1
    images.append(rgb)

fig, ax = plt.subplots(5, 1, figsize=(5, 10))
for i in range(5):
    ax[i].imshow(images[i])
    ax[i].axis('off')
    ax[i].set_title(f'Depth: {i} μm')
plt.tight_layout()
plt.show()
