from scipy import ndimage
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

Prewit1=[[-1, -1, -1],[0,0,0],[1,1,1]]                #Prewit
Roberts1=[[0, 1],[-1, 0]]                              #Roberts
Sobel1=[[-1,0,1],[-2,0,2],[-1,0,1]]                  #Sobel


Prewit2=[[-1,0,1],[-1,0,1],[-1,0,1]]                  #Prewit
Roberts2=[[1, 0],[0, -1]]                              #Roberts
Sobel2=[[1,2,1],[0,0,0],[-1,-2,-1]]                  #Sobel


image = ski.data.camera()
blurred = ski.filters.gaussian(image, 2, channel_axis=True, mode='reflect')

Prewit_1 = ndimage.convolve( blurred, Prewit1 )
Prewit_2 = ndimage.convolve( blurred, Prewit2 )
Prewit = np.sqrt( np.square(Prewit_1) + np.square(Prewit_2))

Roberts_1 = ndimage.convolve( blurred, Roberts1 )
Roberts_2= ndimage.convolve( blurred, Roberts2 )
Roberts = np.sqrt( np.square(Roberts_1) + np.square(Roberts_2))

Sobel_1 = ndimage.convolve( blurred, Sobel1 )
Sobel_2 = ndimage.convolve( blurred, Sobel2 )
Sobel = np.sqrt( np.square(Sobel_1) + np.square(Sobel_2))



edge_roberts = ski.filters.roberts(image)
edge_sobel = ski.filters.sobel(image)
edge_prewit = ski.filters.prewitt(image)

edge_canny = ski.feature.canny(image, sigma=2)


fig, ax = plt.subplots(ncols = 7, sharex=True, sharey=True)

ax[0].imshow(Prewit, cmap='gray')
ax[0].set_title("Prewit")

ax[1].imshow(Roberts, cmap='gray')
ax[1].set_title("Roberts")

ax[2].imshow(Sobel, cmap='gray')
ax[2].set_title("Sobel")


ax[3].imshow(edge_prewit, cmap='gray')
ax[3].set_title("Prewit")

ax[4].imshow(edge_roberts, cmap='gray')
ax[4].set_title("Roberts")

ax[5].imshow(edge_sobel, cmap='gray')
ax[5].set_title("Sobel")


ax[6].imshow(edge_canny, cmap='gray')
ax[6].set_title("canny")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()