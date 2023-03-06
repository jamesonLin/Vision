import matplotlib.pyplot as plt
import skimage as ski

def colorRange(image):
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if (image[i][j][0]/1.8) > image[i][j][1] and (image[i][j][0]/1.8) > image[i][j][2]: #color red
            # if (image[i][j][1]/1.3) > image[i][j][0] and (image[i][j][1]/1.3) > image[i][j][2]: #color green
            # if (image[i][j][2]/1.1) > image[i][j][1] and (image[i][j][2]/1.1) > image[i][j][0]: #color blue
                image[i][j] = image[i][j]
            else:
                image[i][j] = ski.color.rgb2gray(image[i][j])
    return image

def histogram(image):
    hsvImage = ski.color.rgb2hsv(image)
    ravelImage = hsvImage.ravel()
    hue = hsvImage[:, :, 0].ravel()
    # ravelImage = image.ravel()
    # red = image[:, :, 0].ravel()
    # green = image[:, :, 1].ravel()
    # blue = image[:, :, 2].ravel()
    return ravelImage, hue

image = ski.img_as_float(ski.io.imread('mandalas-1084082__340.jpg'))
# image = ski.img_as_float(ski.io.imread('eingang.jpg'))
imageCopy = colorRange(image.copy())

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

ax0.imshow(image)
ax0.set_title("image")

ax1.imshow(imageCopy, cmap='gray')
ax1.set_title("red")

ravelImage, hue = histogram(image)
# ax2.hist(ravelImage, bins = 500, color = 'orange', )
ax2.hist(hue, bins = 500, color = 'red', alpha = 0.5)
# ax2.set_xbound(0, 0.12)
ax2.set_xlabel('color value')
ax2.set_ylabel('pixels')
ax2.legend(['Total', 'Hue'])

ravelImage2, hue2 = histogram(imageCopy)
# ax3.hist(ravelImage2, bins = 500, color = 'orange', )
ax3.hist(hue2, bins = 500, color = 'red', alpha = 0.5)
# ax3.set_xbound(0, 0.12)
ax3.set_xlabel('color value')
ax3.set_ylabel('pixels')
ax3.legend(['Total', 'Hue'])

fig.tight_layout()
plt.show()

