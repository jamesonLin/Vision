from scipy import ndimage
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


image = ski.img_as_float(ski.io.imread('mandalas-1084082__340.jpg'))

tform = ski.transform.SimilarityTransform(scale=1, rotation=np.pi/4, translation=(image.shape[0]/2, -100))
rotated = ski.transform.warp(image.copy(), tform)

tform2 = ski.transform.SimilarityTransform(scale=0.5, translation=(100, 50))
transImage = ski.transform.warp(image, tform2)


src = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
dst = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

tform3 = ski.transform.ProjectiveTransform()
tform3.estimate(src, dst)
stretch = ski.transform.warp(image.copy(), tform3, output_shape=(50, 300))

fig, ax = plt.subplots(ncols = 4, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("original")

ax[1].imshow(rotated, cmap='gray')
ax[1].set_title("draaien")

ax[2].imshow(transImage, cmap='gray')
ax[2].set_title("transleren")

ax[3].imshow(stretch, cmap='gray')
ax[3].set_title("stretchen")


for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()