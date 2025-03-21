import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt


def add_black_border(img, border_size=5):
    """Draw a black border on an RGBA image without changing its size."""
    img[:border_size, :, :3] = 0  # Top border (RGB)
    img[-border_size:, :, :3] = 0  # Bottom border
    img[:, :border_size, :3] = 0  # Left border
    img[:, -border_size:, :3] = 0  # Right border
    return img


axial_img = image.imread('data/figures/926_long/axial_0_overlay.png')[::-1, ::-1]
axial_img = np.rot90(axial_img)
coronal_img = image.imread('data/figures/926_long/coronal_0_overlay.png')[:, ::-1]
sagittal_img = image.imread('data/figures/926_long/sagittal_0_overlay.png')

axial_img = image.imread('data/figures/926_long/axial_0_tomogram.png')[::-1, ::-1]
axial_img = np.rot90(axial_img)
coronal_img = image.imread('data/figures/926_long/coronal_0_tomogram.png')[:, ::-1]
sagittal_img = image.imread('data/figures/926_long/sagittal_0_tomogram.png')

# axial_img = add_black_border(axial_img, border_size=2)
# coronal_img = add_black_border(coronal_img, border_size=2)
# sagittal_img = add_black_border(sagittal_img, border_size=2)

xp_ax, yp_ax, _ = axial_img.shape  # (192, 192, 4)
xp_cor, yp_cor, _ = coronal_img.shape  # (615, 192, 4)
xp_sag, yp_sag, _ = sagittal_img.shape  # (615, 192, 4)

x_ax = np.arange(0, xp_ax, 1)
y_ax = np.arange(0, yp_ax, 1)
Y_ax, X_ax = np.meshgrid(y_ax, x_ax)

x_cor = np.arange(0, xp_cor, 1)
y_cor = np.arange(0, yp_cor, 1)
Y_cor, X_cor = np.meshgrid(y_cor, x_cor)

x_sag = np.arange(0, xp_sag, 1)
y_sag = np.arange(0, yp_sag, 1)
Y_sag, X_sag = np.meshgrid(y_sag, x_sag)

#%%
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# ax.dist = 6.2
ax.view_init(elev=20, azim=240)

ax.plot_surface(X_sag, Y_sag, np.full_like(X_sag, 192), facecolors=sagittal_img,
                rstride=2, cstride=2, antialiased=True, shade=False, rasterized=True)
ax.plot_surface(X_cor, np.full_like(X_cor, 0), Y_cor, facecolors=coronal_img,
                rstride=2, cstride=2, antialiased=True, shade=False, rasterized=True)
ax.plot_surface(np.full_like(X_ax, 0), X_ax, Y_ax, facecolors=axial_img,
                rstride=2, cstride=2, antialiased=True, shade=False, rasterized=True)

ax.set_xlim([0, 615])
ax.set_ylim([0, 615])
ax.set_zlim([0, 615])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()

plt.savefig('data/figures/cuboid_tomogram.svg')
plt.tight_layout()
plt.show()
