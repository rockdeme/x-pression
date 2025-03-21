import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def generate_random_colors(num_colors, hue_range=(0, 1), saturation=0.5, lightness=0.5, min_distance=0.05, seed=None):
    colors = []
    hue_list = []
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed(42)
    while len(colors) < num_colors:
        # generate a random hue value within the specified range
        hue = np.random.uniform(hue_range[0], hue_range[1])
        # check if the hue is far enough away from the previous hue
        if len(hue_list) == 0 or all(abs(hue - h) > min_distance for h in hue_list):
            hue_list.append(hue)
            saturation = saturation
            lightness = lightness
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_code = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            colors.append(hex_code)
    return colors

def get_color_blend(adata, dim=8, hexcodes=None, seed=None, mode='aa', color_first='black'):
    # define PC colors
    if hexcodes is None:
        if dim > 8:
            hexcodes = generate_random_colors(num_colors=dim, min_distance=1 / dim * 0.5)
        else:
            hexcodes = ['#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2']
            if seed is None:
                np.random.seed(len(adata))
            else:
                np.random.seed(seed)
            np.random.shuffle(hexcodes)
    else:
        assert len(hexcodes) >= dim

    # define colormaps
    cmaps = []
    if mode == 'aa':
        for d in range(dim):
            pc_cmap = color_to_color(color_first, hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_aa'][:, d]),
                                           vmax=max(adata.obsm['chr_aa'][:, d]),
                                           value=adata.obsm['chr_aa'][:, d])
            cmaps.append(pc_rgb)

    elif mode == 'pca':
        for d in range(dim):
            pc_cmap = color_to_color(color_first, hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_X_pca'][:, d]),
                                           vmax=max(adata.obsm['chr_X_pca'][:, d]),
                                           value=adata.obsm['chr_X_pca'][:, d])
            cmaps.append(pc_rgb)
    else:
        raise Exception
    cblend = mip_colors(cmaps[0], cmaps[1],)
    if len(cmaps) > 2:
        i = 2
        for cmap in cmaps[2:]:
            cblend = mip_colors(cblend, cmap,)
            i += 1
    return cblend

def color_to_color(first, last):
    # define the colors in the colormap
    colors = [first, last]
    # create a colormap object using the defined colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    return cmap

def get_rgb_from_colormap(cmap, vmin, vmax, value):
    # normalize the value within the range [0, 1]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    value_normalized = norm(value)

    # get the RGBA value from the colormap
    rgba = plt.get_cmap(cmap)(value_normalized)
    # convert the RGBA value to RGB
    # color = tuple(np.array(rgba[:3]) * 255)
    color = np.array(rgba[:, :3])

    return color

def mip_colors(colors_1, colors_2):
    # blend the colors using linear interpolation
    mip_color = []
    for i in range(len(colors_1)):
        r = max(colors_1[i][0], colors_2[i][0])
        g = max(colors_1[i][1], colors_2[i][1])
        b = max(colors_1[i][2], colors_2[i][2])
        mip_color.append((r, g, b))
    return mip_color

#%%
adata = sc.read_h5ad('data/spatial_transcriptomics/h5ads/L2210926_transferred.h5ad')

comps_df = pd.DataFrame(adata.obsm['chr_aa'],
                        columns=[x for x in range(adata.obsm['chr_aa'].shape[1])],
                        index=adata.obs_names)

#%%
with open(r"data/integrated_gradients_umap.pkl", "rb") as input_file:
    grad_df = pickle.load(input_file)

grad_df.index = grad_df['barcode']
grad_df = pd.concat([grad_df, comps_df], axis=1)
cblend = get_color_blend(adata, dim=8, seed=42)
grad_df['colormap'] = cblend

grad_df['idx'] = np.arange(len(grad_df))

grad_df = grad_df.sort_values(by='colormap')
# grad_df = grad_df[comps_df.max(axis=1) > 0.7]

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
ax.axis('off')
ax.scatter(grad_df['UMAP1'], grad_df['UMAP2'],
            c=grad_df['colormap'], s=15, rasterized=True)
# ax.set_title('Feature attributions')
plt.tight_layout()
plt.savefig(f'data/figures/gradients_color_umap.png')
plt.show()

#%%
from sklearn.preprocessing import MinMaxScaler

with open(r"data/integrated_gradients_umap.pkl", "rb") as input_file:
    grad_df = pickle.load(input_file)

patches = []

for i in range(len(grad_df)):
    try:
        with Image.open(f"data/patches_for_umap/patch_{i}.png") as patch:
            patches.append(patch.copy())
    except FileNotFoundError:
        patches.append(None)

patches = np.stack(patches)

min_val, max_val = np.percentile(patches, (1, 99))
patches = np.clip((patches - min_val) / (max_val - min_val), 0, 1)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
scatter = ax.scatter(grad_df["UMAP1"], grad_df["UMAP2"], s=0, rasterized=True)

cmap = cm.get_cmap('gray')

for (x, y, img) in zip(grad_df["UMAP1"], grad_df["UMAP2"], patches):
    if img is not None:
        # Convert image to numpy array

        img_array = np.array(img)

        # scaler = MinMaxScaler()
        # img_array = scaler.fit_transform(img_array)

        img_normalized = img_array / img_array.max()
        img_colored = cmap(img_normalized)
        img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
        imagebox = OffsetImage(img_colored, zoom=0.075)  # Adjust zoom for visibility
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

# ax.set_title('Feature attributions')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'data/figures/gradients_tiles_umap.png')
plt.show()
