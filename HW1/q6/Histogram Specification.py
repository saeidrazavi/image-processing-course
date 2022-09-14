from matplotlib import image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im

# ---------------------------------


def hist_calculator(img: np.ndarray):

    h_r: np.ndarray = cv2.calcHist([img], [0], None, [256], [0, 256])
    h_g: np.ndarray = cv2.calcHist([img], [1], None, [256], [0, 256])
    h_b: np.ndarray = cv2.calcHist([img], [2], None, [256], [0, 256])

    return h_r, h_g, h_b


# -------------------------------
def com_calculator(hr, hg, hb, size):

    commulative_r = np.zeros([256])
    commulative_g = np.zeros([256])
    commulative_b = np.zeros([256])

    for x in range(0, 256, 1):

        commulative_r[x] = round((255)*np.sum(hr[0:x+1]/size))

        commulative_g[x] = round((255)*np.sum(hg[0:x+1]/size))

        commulative_b[x] = round((255)*np.sum(hb[0:x+1]/size))

    return commulative_r, commulative_g, commulative_b


# -------------------------------
def apply_changes(img: np.ndarray, com_r, com_g, com_b):

    [c1, c2, c3] = img.shape

    for y in range(0, c1):

        for g in range(0, c2):

            img[y, g, 0] = com_r[img[y, g, 0]]
            img[y, g, 1] = com_g[img[y, g, 1]]
            img[y, g, 2] = com_b[img[y, g, 2]]


# -----------------------------

def histogram_equalizer(img: np.ndarray, size):

    [h_r, h_g, h_b] = hist_calculator(img)

    [com_r, com_g, com_b] = com_calculator(h_r, h_g, h_b, size)

    apply_changes(img, com_r, com_g, com_b)


# ----------------------------------------------
def closest(list, value):

    list = np.asarray(list)
    idx = (np.abs(list - value)).argmin()
    return list[idx]
# -------------------------------


def hist_specifier(im1, im2):

    [c1, c2, c3] = im1.shape
    size1 = int(c1)*int(c2)

    [c11, c22, c33] = im2.shape
    size2 = int(c11)*int(c22)

    cnew_r = np.zeros([256])

    cnew_g = np.zeros([256])

    cnew_b = np.zeros([256])

    [h1_r, h1_g, h1_b] = hist_calculator(im1)

    [h2_r, h2_g, h2_b] = hist_calculator(im2)

    [com1_r, com1_g, com1_b] = com_calculator(h1_r, h1_g, h1_b, size1)

    [com2_r, com2_g, com2_b] = com_calculator(h2_r, h2_g, h2_b, size2)

    for x in range(0, 256):

        temp = closest(com2_r, com1_r[x])
        indice = np.nonzero(com2_r == temp)[0][0]
        cnew_r[int(com1_r[x])] = indice

        temp1 = closest(com2_b, com1_b[x])
        indice1 = np.nonzero(com2_b == temp1)[0][0]
        cnew_b[int(com1_b[x])] = indice1

        temp2 = closest(com2_g, com1_g[x])
        indice2 = np.nonzero(com2_g == temp2)[0][0]
        cnew_g[int(com1_g[x])] = indice2

    return cnew_r, cnew_g, cnew_b


# ----------------------------------------------
imm = cv2.imread("dark.jpg")
im1 = np.array(imm)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im1 = np.array(im1)

immm = cv2.imread("pink.jpg")
im2 = np.array(immm)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)


[c1, c2, c3] = im1.shape
size = int(c1)*int(c2)

im3 = im1.copy()


histogram_equalizer(im3, size)


[c1, c2, c3] = hist_specifier(im1, im2)

apply_changes(im3, c1, c2, c3)

h_r_sp, bins_rs = np.histogram(im3[:, :, 0], range(0, 256))
h_g_sp, bins_gs = np.histogram(im3[:, :, 1], range(0, 256))
h_b_sp, bins_bs = np.histogram(im3[:, :, 2], range(0, 256))


plt.figure(figsize=(8, 6))
plt.plot(h_r_sp, color='red')
plt.plot(h_g_sp, color='green')
plt.plot(h_b_sp, color='blue')
plt.title('histogram of final image ')
plt.savefig('res10.png')
plt.show()


plt.imshow(im3)
plt.title('final image')

plt.show()

data = im.fromarray(np.uint8(im3))
data.save('res11.jpg')  # save the result
