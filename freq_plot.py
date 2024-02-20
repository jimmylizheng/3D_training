import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def plot_and_save_frequency_domain(image_path, output_folder):
    # Read the image
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    channel = 'r'
    # Read the image
    img = cv2.imread(image_path)

    # Extract the specified color channel
    if channel == 'r':
        img_channel = img[:, :, 0]  # Red channel
    elif channel == 'g':
        img_channel = img[:, :, 1]  # Green channel
    elif channel == 'b':
        img_channel = img[:, :, 2]  # Blue channel
        
    f_transform = np.fft.fft2(img_channel)
    
    # Apply FFT to the image
    # f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Optionally, take the logarithm of the magnitude spectrum for better visualization
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)

    # Plot the magnitude spectrum
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum_log, cmap='gray')
    plt.title('Frequency Domain')
    plt.colorbar()
    
    # Save the plotted image
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_frequency_domain.png")
    plt.savefig(output_path)
    plt.close()

# Folder containing images
# input_folder = "images_high/"
input_folder = "images_low_30/"
# input_folder = "images/"
output_folder = "frequency_domain_images_lf_red/"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through image files, plot frequency domain, and save images
for jpg_file in os.listdir(input_folder):
    if jpg_file.endswith(".jpg"):
        image_path = os.path.join(input_folder, jpg_file)
        file_id = int(os.path.splitext(os.path.basename(image_path))[0])
        if file_id<10:
            plot_and_save_frequency_domain(image_path, output_folder)
