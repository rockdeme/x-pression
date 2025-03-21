import cv2
import openslide
import numpy as np
import scanpy as sc
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, translate, rotate

def gene_symbol_to_ensembl_id(adata, gene_symbol_col='gene_symbols', ensembl_col='gene_ids'):
    gene_symbols_copy = list(adata.var_names)
    gene_ids_copy = list(adata.var[ensembl_col])
    # change var_names to gene_ids
    adata.var_names = gene_ids_copy
    adata.var.index = gene_ids_copy
    # assign the copied gene symbols back to the 'gene_symbols' column
    adata.var[gene_symbol_col] = gene_symbols_copy
    return adata


def ensembl_id_to_gene_symbol(adata, gene_symbol_col='gene_symbols', ensembl_col='gene_ids'):
    gene_ids_copy = list(adata.var_names)
    gene_symbols_copy = list(adata.var[gene_symbol_col])
    # change var_names to gene_ids
    adata.var_names = gene_symbols_copy
    adata.var.index = gene_symbols_copy
    # assign the copied gene symbols back to the 'gene_symbols' column
    adata.var[ensembl_col] = gene_ids_copy
    return adata


def density_scatter(x, y, s=3, cmap='viridis', ax=None):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    nx, ny, z = x[idx], np.array(y)[idx], z[idx]
    x_list = nx.tolist()
    y_list = ny.tolist()
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    return plt.scatter(x_list, y_list, alpha=1, c=z, s=s, zorder=2, cmap=cmap)


def segment_tissue(img, scale=1.05, l=50, h=200):

    def detect_contour(img, low, high):
        img = img * 255
        img = np.uint8(img)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        cnt_info = []
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            cnt_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
        cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
        cnt = cnt_info[0][0]
        return cnt

    def scale_contour(cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)
        return cnt_scaled

    cnt = detect_contour(img, l, h)
    cnt_enlarged = scale_contour(cnt, scale)
    binary = np.zeros(img.shape[0:2])
    cv2.drawContours(binary, [cnt_enlarged], -1, 1, thickness=-1)
    img[binary == 0] = 1

    return img, binary, cnt_enlarged

def plot_adatas(adatas, rows=5, cols=5, color=None, cmap='viridis', alpha=1, size=4):
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, ad in enumerate(adatas):
        sc.pl.spatial(ad, color=color, size=1.5, alpha=alpha,
                      ax=ax[idx], show=False, cmap=cmap)
        ax[idx].set_title(ad.obs['sample'].values[0])
    plt.tight_layout()


def parse_visiopharm_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    # Get the root element
    root = tree.getroot()
    # Find all Object elements that contain Vertices
    objects_with_vertices = root.findall(".//Object[Vertices]")
    # Initialize a list to store all polygons
    polygons = []
    types = []
    # Loop through each object with vertices
    for obj in objects_with_vertices:
        vertices = []
        for vertex in obj.findall("Vertices/Vertex"):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            vertices.append((x, y))

        # Create a Shapely polygon from the vertices and add to the list
        polygon = Polygon(vertices)
        polygons.append(polygon)
        types.append(obj.attrib['Type'])
    # Output the number of polygons and optionally the polygons themselves
    # print(f"Number of polygons: {len(polygons)}")
    # for i, poly in enumerate(polygons[:5]):  # Print the first 5 polygons as an example
    #     print(f"Polygon {i + 1}: {poly}")

    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf['type'] = types
    return gdf


def map_visium_to_visiopharm(adata, gdf, slide_path, visium_bbox_file, rot_spots=-90):
    source = openslide.OpenSlide(slide_path)
    visium_bbox = pd.read_csv(visium_bbox_file, index_col=0)

    props = source.properties
    thumbnail = source.get_thumbnail((800, 800))
    # plt.imshow(thumbnail)
    # plt.show()

    # Convert to nanometers
    mpp_x = float(props['openslide.mpp-x'])  # microns per pixel (x)
    mpp_y = float(props['openslide.mpp-y'])  # microns per pixel (y)

    width_pixels = float(props['openslide.level[0].width'])
    height_pixels = float(props['openslide.level[0].height'])

    offset_x = float(props['hamamatsu.XOffsetFromSlideCentre'])
    offset_y = float(props['hamamatsu.YOffsetFromSlideCentre'])

    offset_x_mm = offset_x / 1000000  # this is in nm
    offset_y_mm = offset_y / 1000000  # this is in nm

    # convert scan area dimensions to nanometers
    width_mm = width_pixels * mpp_x / 1000  # convert microns to mms
    height_mm = height_pixels * mpp_y / 1000  # convert microns to mms

    scale_factor = 4  # coming from the downsample value as we used the level 2 image

    visium_bbox.columns = [x.strip() for x in visium_bbox.columns]
    bbox_height = np.max(visium_bbox['Y']) - np.min(visium_bbox['Y'])
    bbox_width = np.max(visium_bbox['X']) - np.min(visium_bbox['X'])

    spot_coords = adata.obsm['spatial'].copy()
    spots = [Point(xy) for xy in zip(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1])]
    spots = [rotate(p, rot_spots, origin=(0, 0)) for p in spots]
    spots = [translate(p, yoff=bbox_height) for p in spots]

    # translate coords by the bounding box
    spots = [translate(p, xoff=np.min(visium_bbox['X']), yoff=np.min(visium_bbox['Y'])) for p in spots]

    # scale them to mm on the original SCAN AREA coordinate system
    spots = [scale(p, xfact=4, yfact=4, origin=(0, 0)) for p in spots]  # downscaling from pyramids
    spots = [scale(p, xfact=mpp_x, yfact=mpp_y, origin=(0, 0)) for p in spots]  # mpp values
    spots = [scale(p, xfact=0.001, yfact=0.001, origin=(0, 0)) for p in spots]  # um to mm

    # we need the center point of the SCAN AREA
    spots = [scale(p, xfact=1, yfact=-1, origin=(width_mm / 2, height_mm / 2)) for p in spots]
    spots = [translate(p, xoff=-width_mm / 2, yoff=-height_mm / 2) for p in spots]
    spots = [translate(p, xoff=offset_x_mm, yoff=-offset_y_mm) for p in spots]

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # gdf.plot(column='type', ax=ax)
    # ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    # gpd.GeoDataFrame(geometry=spots).plot(markersize=1, ax=ax)
    # ax.set_axisbelow(True)
    # plt.show()

    spots_gdf = gpd.GeoDataFrame(geometry=spots)
    spots_gdf.index = adata.obs_names
    return spots_gdf



def spatial_plot(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis', subplot_size=4,
                 alpha_img=1.0, share_range=True, suptitle=None, wspace=0.5, hspace=0.5, colorbar_label=None,
                 colorbar_aspect=20, colorbar_shrink=0.7, colorbar_outline=True, alpha_blend=False, k=15, x0=0.5,
                 suptitle_fontsize=20, suptitle_y=0.99, topspace=None, bottomspace=None, leftspace=None,
                 rightspace=None, facecolor='white', wscale=1,
                 **kwargs):

    from pandas.api.types import is_categorical_dtype

    if share_range:
        if var in list(adata.var_names):
            gene_index = adata.var_names.get_loc(var)
            expression_vector = adata[:, gene_index].X.toarray().flatten()
            if not is_categorical_dtype(adata.obs[var].dtype):
                vmin = np.percentile(expression_vector, 0.2)
                vmax = np.percentile(expression_vector, 99.8)
                if alpha_blend:
                    vmin = np.min(expression_vector)
                    vmax = np.max(expression_vector)
        else:
            if not is_categorical_dtype(adata.obs[var].dtype):
                vmin = np.percentile(adata.obs[var], 0.2)
                vmax = np.percentile(adata.obs[var], 99.8)
                if alpha_blend:
                    vmin = np.min(adata.obs[var])
                    vmax = np.max(adata.obs[var])

    fig, ax = plt.subplots(rows, cols, figsize=(cols * subplot_size * wscale, rows * subplot_size), sharey=True)
    ax = np.array([ax])

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
        for a in ax:
            a.axis('off')
    else:
        ax.axis('off')
        ax = list([ax])

    # Plot for each sample
    for idx, s in enumerate(adata.obs[sample_col].cat.categories):
        ad = adata[adata.obs[sample_col] == s].copy()

        if not share_range:
            if var in list(ad.var_names):
                gene_index = ad.var_names.get_loc(var)
                expression_vector = ad[:, gene_index].X.toarray().flatten()
                if not is_categorical_dtype(adata.obs[var].dtype):
                    vmin = np.percentile(expression_vector, 0.2)
                    vmax = np.percentile(expression_vector, 99.8)
                    if alpha_blend:
                        vmin = np.min(expression_vector)
                        vmax = np.max(expression_vector)
            else:
                if not is_categorical_dtype(adata.obs[var].dtype):
                    vmin = np.percentile(ad.obs[var], 0.2)
                    vmax = np.percentile(ad.obs[var], 99.8)
                    if alpha_blend:
                        vmin = np.min(ad.obs[var])
                        vmax = np.max(ad.obs[var])

        if alpha_blend:
            if var in adata.var_names:
                gene_idx = adata.var_names.get_loc(var)
                values = ad[:, gene_idx].X.toarray().flatten()
            else:
                values = ad.obs[var].values

            # Normalize values to [0, 1] to use as alpha values
            norm_values = (values - vmin) / (vmax - vmin + 1e-10)
            norm_values = 1 / (1 + np.exp(-k * (norm_values - x0)))

            alpha_df = pd.DataFrame({'v': values, 'a': norm_values})
            alpha_df = alpha_df.sort_values(by='v', ascending=True)
            kwargs['alpha'] = list(alpha_df['a'])  # Pass the alpha values to scanpy plot

        if share_range:
            with mpl.rc_context({'axes.facecolor': facecolor}):
                sc.pl.spatial(ad, color=var, size=1.5, library_id=s,
                              ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img,
                              vmin=vmin, vmax=vmax, colorbar_loc=None, **kwargs)
                print(vmin, vmax)
        else:
            with mpl.rc_context({'axes.facecolor': facecolor}):
                sc.pl.spatial(ad, color=var, size=1.5, library_id=s,
                              ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, colorbar_loc=None,
                              **kwargs)

        if title is not None:
            ax[idx].set_title(title[idx])

    # Adjust colorbars only if ranges are shared
    if share_range:
        # Add colorbar only for the last plot in each row
        for r in range(rows):
            last_in_row_idx = (r + 1) * cols - 1  # Last subplot in each row

            # doesnt work if i nthe last row we have less samples so fix this at some point
            sc_img = ax[last_in_row_idx].collections[0]

            colorbar = fig.colorbar(sc_img, ax=ax[last_in_row_idx], shrink=colorbar_shrink, aspect=colorbar_aspect,
                                    format="%.3f")
            colorbar.outline.set_visible(colorbar_outline)
            colorbar.set_label(colorbar_label)
    else:
        for r in range(len(ax)):

            # doesnt work if i nthe last row we have less samples so fix this at some point
            sc_img = ax[r].collections[0]

            colorbar = fig.colorbar(sc_img, ax=ax[r], shrink=colorbar_shrink, aspect=colorbar_aspect,
                                    format="%.1f")
            colorbar.outline.set_visible(colorbar_outline)
            colorbar.set_label(colorbar_label)

    # Adjust the gap between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=topspace, bottom=bottomspace,
                                                                           left=leftspace, right=rightspace)

    if suptitle:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)
    # plt.tight_layout()


def matrixplot(df, figsize = (7, 5), num_genes=5, hexcodes=None,
                seed: int = None, scaling=True, reorder_comps=True, reorder_obs=True, comps=None, flip=True,
                colorbar_shrink: float = 0.5, colorbar_aspect: int = 20, cbar_label: str = None,
                dendrogram_ratio=0.05, xlabel=None, ylabel=None, fontsize=10, title=None, cmap=None,
                color_comps=True, xrot=0, ha='right', select_top=False, fill_diags=False, dendro_treshold=None,
                **kwrgs):
    # SVG weights for each compartment

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.cluster.hierarchy import dendrogram
    import seaborn as sns
    import chrysalis as ch
    from scipy.cluster.hierarchy import linkage, leaves_list
    import matplotlib.ticker as ticker

    dim = df.shape[0]
    hexcodes = ch.utils.get_hexcodes(hexcodes, dim, seed, 1)

    if cmap is None:
        cmap = sns.diverging_palette(45, 340, l=55, center="dark", as_cmap=True)

    if comps:
        df = df.T[comps]
        df = df.T
        hexcodes = [hexcodes[x] for x in comps]

    if scaling:
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        df = df.dropna(axis=1)
    if reorder_obs:
        z = linkage(df.T, method='ward')
        order = leaves_list(z)
        df = df.iloc[:, order]
    if reorder_comps:
        z = linkage(df, method='ward')
        order = leaves_list(z)
        df = df.iloc[order, :]
        hexcodes = [hexcodes[i] for i in order]
    if select_top:
        # get top genes for each comp
        genes_dict = {}
        for idx, row in df.iterrows():
            toplist = row.sort_values(ascending=False)
            genes_dict[idx] = list(toplist.index)[:5]
        selected_genes = []
        for v in genes_dict.values():
            selected_genes.extend(v)

        plot_df = df[selected_genes]
    else:
        plot_df = df

    if flip:
        plot_df = plot_df.T
        d_orientation = 'right'
        d_pad = 0.00
    else:
        d_orientation = 'top'
        d_pad = 0.00

    fig, ax = plt.subplots(figsize=figsize)

    if fill_diags:
        np.fill_diagonal(plot_df.values, 0)

    # Create the heatmap but disable the default colorbar
    sc_img = sns.heatmap(plot_df.T, ax=ax, cmap=cmap,
                         center=0, cbar=False, zorder=2, **kwrgs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, ha=ha)
    # Create a divider for axes to control the placement of dendrogram and colorbar
    divider = make_axes_locatable(ax)

    # Dendrogram axis - append to the left of the heatmap
    ax_dendro = divider.append_axes(d_orientation, size=f"{dendrogram_ratio * 100}%", pad=d_pad)

    # Plot the dendrogram on the left
    dendro = dendrogram(z, orientation=d_orientation, ax=ax_dendro, no_labels=True, color_threshold=0,
                        above_threshold_color='black')
    if flip:
        ax_dendro.invert_yaxis()  # Invert to match the heatmap

    # Remove ticks and spines from the dendrogram axis
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    if dendro_treshold:
        ax_dendro.axhline(dendro_treshold, color='grey', dashes=(5, 5))


    # Set tick label colors
    if flip:
        ticklabels = ax.get_yticklabels()
    else:
        ticklabels = ax.get_xticklabels()

    if color_comps:
        for idx, t in enumerate(ticklabels):
            t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))

    # Create the colorbar using fig.colorbar with shrink and aspect
    colorbar = fig.colorbar(sc_img.get_children()[0], ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=0.02)
    colorbar.locator = ticker.MaxNLocator(nbins=5)  # Adjust the number of ticks
    colorbar.update_ticks()

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if d_orientation == 'top':
        ax_dendro.set_title(title)
    else:
        ax.set_title(title)

    # Set the colorbar label
    if cbar_label:
        colorbar.set_label(cbar_label)
    plt.tight_layout()
