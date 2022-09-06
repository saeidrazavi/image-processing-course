from matplotlib import image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im

# --------------------------------------------------


def editor(im1: np.ndarray):
    im2 = im1.copy()
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2HSV)

    alpha1 = 0.001

    alpha2 = 0.1

    im2[:, :, 1] = 255 / \
        np.log10(1+255*alpha1)*np.log10(1+alpha1*im2[:, :, 1])

    im2[:, :, 2] = 255 / \
        np.log10(1+255*alpha2)*np.log10(1+alpha2*im2[:, :, 2])

    im2 = cv2.cvtColor(im2, cv2.COLOR_HSV2RGB)

    return im2


# explnation : using logarithm function with paramter g=0.5 to make dark pixels brighter
# --------------------------------------
img = plt.imread('Enhance2.jpg')  # read image
img1 = np.array(img)

[c1, c2, c3] = img1.shape
size = int(c1)*int(c2)


final_img = editor(img1)  # pass image to editor function

# ---------------------------

data = im.fromarray(final_img)
data.save('res02.jpg')  # save the result

# ----------------------------------

plt.imshow(final_img)  # showing images
plt.show()
