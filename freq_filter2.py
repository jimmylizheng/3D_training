# freq_filter implemented without gaussian blur

import cv2
import numpy as np
from matplotlib import pyplot as plt

def low_pass_filter(image, cutoff_freq):
    # Apply FFT to the image
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a mask for the low-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_freq  # Radius of the circular mask
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    # Apply the mask to the frequency domain
    f_transform_shifted = f_transform_shifted * mask

    # Inverse FFT to get back the filtered image
    img_filtered = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted)).real

    return img_filtered

def low_pass_filter_color(image, cutoff_freq):
    if cutoff_freq > 300: # for cutoff that is high enough, do not do the filtering
        return image
    
    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply the low-pass filter to each color channel
    b_filtered = low_pass_filter(b, cutoff_freq)
    g_filtered = low_pass_filter(g, cutoff_freq)
    r_filtered = low_pass_filter(r, cutoff_freq)

    # Merge the filtered color channels back into a BGR image
    img_filtered = cv2.merge([b_filtered, g_filtered, r_filtered])

    return img_filtered

# Load the color image
image_color = cv2.imread('images/00001.jpg')

# Apply the low-pass filter with a cutoff frequency to the color image
cutoff_frequency = 230
lpf_image = low_pass_filter_color(image_color, cutoff_frequency) # low pass filter image
hpf_image = image_color - lpf_image

# Save the filtered color image as a JPEG file
# cv2.imwrite('low_filtered_image.jpg', lpf_image)
# cv2.imwrite('high_filtered_image.jpg', hpf_image)
cv2.imwrite('high_filtered_image.jpg', lpf_image)
cv2.imwrite('low_filtered_image.jpg', hpf_image)