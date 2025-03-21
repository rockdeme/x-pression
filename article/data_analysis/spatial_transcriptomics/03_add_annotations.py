import  numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_analysis.utils import parse_visiopharm_xml, map_visium_to_visiopharm, plot_adatas


# map annotations
sample_folder = 'data/spatial_transcriptomics/h5ads'
sample_suffix = 'preprocessed'
xml_folder = '/mnt/f/10X_visium_slides/covid_lung_annotations'
bbox_folder = '/mnt/f/10X_visium_slides'

samples = glob(sample_folder + f'/*{sample_suffix}*.h5ad')
xmls = glob(f'{xml_folder}/*.xml')
bboxes = glob(f'{bbox_folder}/*/*/*.csv')
slides = glob(f'{bbox_folder}/*/*.ndpi')

metadata_df = pd.read_csv('data_analysis/metadata.csv', index_col=0)

sample_adatas = {}
for s in tqdm(samples):
    pass
    adata = sc.read_h5ad(s)
    label = adata.obs['sample'].values[0]
    meta_row = metadata_df.loc[label]

    xml_file = [x for x in xmls if meta_row['whole_slide_image'][:-5] in x]
    assert len(xml_file) == 1
    xml_file = xml_file[0]

    slide = [x for x in slides if meta_row['whole_slide_image'] in x]
    assert len(slide) == 1
    slide = slide[0]

    bbox = [x for x in bboxes if meta_row['whole_slide_bbox'] in x]
    assert len(bbox) == 1
    bbox = bbox[0]

    annots_gdf = parse_visiopharm_xml(xml_file)

    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    annots_gdf.plot(column='type', ax=ax, alpha=0.5)
    annots_gdf.boundary.plot(color='black', ax=ax)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    plt.title(xml_file)
    ax.set_axisbelow(True)
    plt.tight_layout()
    # plt.savefig(f'{xml_file[:-4]}.png')
    plt.show()

    spots_gdf = map_visium_to_visiopharm(adata, annots_gdf, slide, bbox)

    # ensure that the spots map to the smallest polygon
    annots_gdf['area'] = annots_gdf.geometry.area
    annots_gdf = annots_gdf.sort_values(by='area', ascending=True)
    spots_gdf['assigned_polygon'] = None
    for idx, polygon in annots_gdf.iterrows():
        points_in_polygon = spots_gdf[spots_gdf.geometry.within(polygon.geometry) &
                                      spots_gdf['assigned_polygon'].isna()]
        spots_gdf.loc[points_in_polygon.index, 'assigned_polygon'] = polygon['type']

    # add unlabeled spots that are in the tissue
    adata.obs['annotation'] = spots_gdf['assigned_polygon']
    unlabeled = ['unlabeled' if x == True else np.nan
                 for x in adata.obs['annotation'].isna() & adata.obs['in_tissue'] == 1]
    new_vals = []
    for annot, val in zip(adata.obs['annotation'], unlabeled):
        if annot == None:
            new_vals.append(val)
        else:
            new_vals.append(annot)
    adata.obs['annotation'] = new_vals


    annotation_names = {'0': 'unlabeled',
                        '1': 'lung_parenchyma',
                        '2': 'bronchiole_lumen',
                        '3': 'blood_vessel_lumen',
                        '4': 'inflammation',
                        '5': 'atelectasis',
                        '6': 'prominent_inflammation',
                        'unlabeled': 'unlabeled',
                        np.nan: np.nan}

    adata.obs['annotation'] = [annotation_names[x] for x in adata.obs['annotation']]
    categs = list(annotation_names.values())[1:-1]
    adata.obs['annotation'] = adata.obs['annotation'].astype('category')
    adata.obs['annotation'] = adata.obs['annotation'].cat.set_categories(categs)

    sc.pl.spatial(adata, color=['annotation'])
    plt.show()

    adata.write(f'{sample_folder}/{label}_{sample_suffix}.h5ad')

    sample_adatas[label] = adata

#%%

sample_adatas = {}
for s in tqdm(samples):
    pass
    adata = sc.read_h5ad(s)
    label = adata.obs['sample'].values[0]
    sample_adatas[label] = adata

plot_adatas(sample_adatas.values(), color='annotation', alpha=1, rows=5, cols=6, size=5)
plt.show()
