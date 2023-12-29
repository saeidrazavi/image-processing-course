import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image as im

# --------------------------------


def align(im1, im2, p1, p3):
    im1 = np.array(im1)
    [c1, c2, c3] = im1.shape

    # difference in x position for left eyes of two images
    delta_x = p3[0]-p1[0]
    # difference in y position for left eyes of two images
    delta_y = p3[1]-p1[1]

    im2 = np.roll(im2, int(-delta_y), axis=0)  # shift image up/down

    im2 = np.roll(im2, int(-delta_x), axis=1)  # shift image right/left

    if(delta_x > 0):

        im1 = im1[:, :int(c2-delta_x), :]
        # crop image after shifting image
        im2 = im2[:, :int(c2-delta_x), :]

    else:

        im1 = im1[:, int(-delta_x):, :]
        # crop image after shifting image
        im2 = im2[:, int(-delta_x):, :]

    if(delta_y > 0):

        im1 = im1[:int(c1-delta_y), :, :]
        # crop image after shifting image
        im2 = im2[:int(c1-delta_y), :, :]

    else:

        im1 = im1[int(-delta_y):, :, :]
        # crop image after shifting image
        im2 = im2[int(-delta_y):, :, :]

    return im1, im2

# --------------------------------


def freq_to_time_domain(r, g, b, image):

    image1 = np.array(image)

    [c1, c2, c3] = image1.shape

    time_domain_image = np.zeros([c1, c2, c3])

    r_ifft = (fft_inverse_cal(r)).astype('uint8')
    g_ifft = (fft_inverse_cal(g)).astype('uint8')
    b_ifft = (fft_inverse_cal(b)).astype('uint8')

    time_domain_image = mergging(
        r_ifft, g_ifft, b_ifft, image).clip(0, 255)

    return time_domain_image

# -----------------------------------


def low_guassian(n1: int, n2: int, sigma):

    c1 = (n1-1)//2

    c2 = (n2-1)//2

    matrix = np.zeros([n1, n2])

    for x in range(0, n1):
        for y in range(0, n2):

            matrix[x, y] = np.exp(-((x-c1)**2+(y-c2)**2)/2/(sigma**2))

    plt.imsave(f'res26-lowpass-{sigma}.jpg', matrix)

    return matrix

# ------------------------------


def high_guassian(n1: int, n2: int, sigma):

    c1 = (n1-1)//2

    c2 = (n2-1)//2

    matrix = np.zeros([n1, n2], dtype='float32')

    for x in range(0, n1):
        for y in range(0, n2):

            matrix[x, y] = np.exp(-((x-c1)**2+(y-c2)**2)/2/(sigma**2))

    matrix = 1-matrix

    plt.imsave(f'res25-highpass-{sigma}.jpg', matrix)

    return matrix

# --------------------------------


def filtering(r, g, b, mask2):

    eq1_r = mask2*r
    eq1_g = mask2*g
    eq1_b = mask2*b

    return eq1_r, eq1_g, eq1_b

# --------------------------------


def mergging(r, g, b, image1):

    merged_image = np.zeros_like(image1)

    r_abs = np.abs(r)
    g_abs = np.abs(g)
    b_abs = np.abs(b)

    merged_image = cv2.merge((r_abs, g_abs, b_abs))

    return merged_image

# -------------------------------


def fft_cal(image):

    dft = np.fft.fft2(image)

# # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    return dft_shift

# ---------------------------------


def fft_inverse_cal(dft):

    ishift = np.fft.ifftshift(dft)

    ifft = np.fft.ifft2(ishift)

    return np.real(ifft)

# ------------------------------------
# Read first not_aligned image


image1 = cv2.imread("res19-near.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1 = np.array(image1[:, :, :])
[c1, c2, c3] = image1.shape
plt.imshow(image1)
(p1, p2) = plt.ginput(2)
plt.close()


# read second not aligned image
image2 = cv2.imread("res20-far.jpeg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = np.array(image2)
image2 = cv2.resize(image2, dsize=(c2, c1), interpolation=cv2.INTER_CUBIC)
plt.imshow(image2)
(p3, p4) = plt.ginput(2)
plt.close()

#----------------------------------------- aligning
im1, im2 = align(image1, image2, p1, p3)
im1 = np.array(im1)
im2 = np.array(im2)

# ---------------------------------------- store aligned images
data = im.fromarray(np.uint8(im2))
data.save('res22-far.jpg')  # save the result

data = im.fromarray(np.uint8(im1))
data.save('res21-near.jpg')  # save the result

# -------------------------------

image1 = cv2.imread("res21-near.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1_grayscale = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image1 = np.array(image1[:, :, :])
[c1, c2, c3] = image1.shape

# ------------------------------------

image2 = cv2.imread("res22-far.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2_grayscale = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
image2 = np.array(image2[:, :, :])
[c1, c2, c3] = image2.shape

# ------------------------------------

r1, g1, b1 = cv2.split(image1)

r1_dft = fft_cal(r1)
g1_dft = fft_cal(g1)
b1_dft = fft_cal(b1)

# -----------------------------------
r2, g2, b2 = cv2.split(image2)

r2_dft = fft_cal(r2)
g2_dft = fft_cal(g2)
b2_dft = fft_cal(b2)
# -----------------------------------

low = low_guassian(image1.shape[0], image1.shape[1], 20)
plt.imsave('lowpassfilter.jpg', low)

high = high_guassian(image1.shape[0], image1.shape[1], 65)
plt.imsave('highpassfilter.jpg', high)
# ------------------------------------

r_filtered, g_filtered, b_filtered = filtering(r1_dft, g1_dft, b1_dft, high)
image_high = freq_to_time_domain(r_filtered, g_filtered, b_filtered, image1)

r_filtered2, g_filtered2, b_filtered2 = filtering(r2_dft, g2_dft, b2_dft, low)
image_low = freq_to_time_domain(r_filtered2, g_filtered2, b_filtered2, image2)

# -------------------------------------add highpassed and lowpassed image

final_image = (1*image_high+1*image_low)


# ---------------------------------------saving results :

# showing fft of aligned images :

fft_image_1 = mergging(r1_dft, g1_dft, b1_dft, image1)
mag = np.abs(fft_image_1)
spec = np.log(mag)/20
plt.imsave('res23-dft-near.jpg', spec.clip(0, 1).astype(np.float32))


fft_image_2 = mergging(r2_dft, g2_dft, b2_dft, image1)
mag2 = np.abs(fft_image_2)
spec2 = np.log(mag2)/20
plt.imsave('res24-dft-far.jpg', spec2.clip(0, 1).astype(np.float32))


res27_highpassed = mergging(r_filtered, g_filtered, b_filtered, image1)
mag3 = np.abs(res27_highpassed)
spec3 = np.log(mag3)/20
plt.imsave('res27-highpassed.jpg', spec3.clip(0, 1).astype(np.float32))


res28_lowpassed = mergging(r_filtered2, g_filtered2, b_filtered2, image1)
mag4 = np.abs(res28_lowpassed)
spec4 = np.log(mag4)/20
plt.imsave('res28-lowpassed.jpg', spec4.clip(0, 1).astype(np.float32))


res29_hybrid = np.fft.fft2(final_image, axes=(0, 1))
res29_hybrid = np.fft.fftshift(res29_hybrid)
plt.imsave('res29-hybrid.jpg', res29_hybrid.clip(0, 1).astype(np.float32))


final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

res30_hybrid_near = np.real(cv2.resize(
    final_image_bgr, (int(0.8*c2), int(0.8*c1)), interpolation=cv2.INTER_AREA))
res31_hybrid_far = np.real(cv2.resize(final_image_bgr, (int(
    0.12*c2), int(0.12*c1)), interpolation=cv2.INTER_AREA))

cv2.imwrite('res30-hybrid-near.jpg', res30_hybrid_near)
cv2.imwrite('res31-hybrid-far.jpg', res31_hybrid_far)
