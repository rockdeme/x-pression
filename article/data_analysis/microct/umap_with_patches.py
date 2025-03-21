import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

df_umap = pd.read_pickle("df_umap.pkl")
patches = []
for i in range(len(df_umap)):
    try:
        patch = Image.open(f"patches_for_umap/patch_{i}.png")
        patches.append(patch)
    except FileNotFoundError:
        patches.append(None)

fig, ax = plt.subplots(figsize=(15, 12), dpi=150)

scatter = ax.scatter(df_umap["UMAP1"], df_umap["UMAP2"], c=df_umap["class_idx"], cmap="tab10", s=0)

for (x, y, img) in zip(df_umap["UMAP1"], df_umap["UMAP2"], patches):
    if img is not None:
        imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom for visibility
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
plt.colorbar(scatter, label="Class")

ax.set_title("Attributions after PCA with Patches", fontsize=16)
ax.set_xlabel("UMAP1", fontsize=14)
ax.set_ylabel("UMAP2", fontsize=14)

plt.show()
