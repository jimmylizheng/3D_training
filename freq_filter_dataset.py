# freq_filter implemented without gaussian blur
import os
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

    return image - img_filtered


root=""
input_folder = "images/"
jpg_files = [file for file in sorted(os.listdir(root + input_folder)) if file.endswith(".jpg")]

blurring_size = 30
low_output_folder = root + f"images_low_{blurring_size}/"
if not os.path.exists(low_output_folder):
    os.makedirs(low_output_folder)
    
high_output_folder = root + "images_high/"
if not os.path.exists(high_output_folder):
    os.makedirs(high_output_folder)
    
for jpg_file in jpg_files:
    input_image_path = os.path.join(root + input_folder, jpg_file)
    img = cv2.imread(input_image_path)
    # Apply low-pass filter using Gaussian blur
    low_pass_img = low_pass_filter_color(img, blurring_size)

    # Save the low-pass filtered image to the output folder
    output_image_path = os.path.join(low_output_folder, f"{jpg_file[:-4]}.jpg")
    cv2.imwrite(output_image_path, low_pass_img)
    
    
    # Apply high-pss filter complement with the Gaussian blur
    high_pass_img = img - low_pass_img
    output_image_path = os.path.join(high_output_folder, f"{jpg_file[:-4]}.jpg")
    cv2.imwrite(output_image_path, high_pass_img)
    