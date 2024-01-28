import os
import cv2
import numpy as np

# root = "/Users/yizhen91/Document/UCR/3d_volumetric_video_streaming/gaussian-splatting/tandt/train/"
root=""
input_folder = "images/"
jpg_files = [file for file in sorted(os.listdir(root + input_folder)) if file.endswith(".jpg")]

blurring_size = 31

low_output_folder = root + f"images_low_{blurring_size}/"
if not os.path.exists(low_output_folder):
    os.makedirs(low_output_folder)
    
high_output_folder = root + "images_high/"
if not os.path.exists(high_output_folder):
    os.makedirs(high_output_folder)
    
for jpg_file in jpg_files:
    input_image_path = os.path.join(root + input_folder, jpg_file)
    img = cv2.imread(input_image_path)

#     # Apply high-pass filter using Laplacian operator
#     high_pass_img = cv2.Laplacian(img, cv2.CV_64F)
#     high_pass_img = np.uint8(np.abs(high_pass_img))

#     # Save the high-pass filtered image to the output folder
#     output_image_path = os.path.join(high_output_folder, f"{jpg_file[:-4]}.jpg")
#     cv2.imwrite(output_image_path, high_pass_img)
    
    
    # Apply low-pass filter using Gaussian blur
    low_pass_img = cv2.GaussianBlur(img, (blurring_size, blurring_size), 0)

    # Save the low-pass filtered image to the output folder
    output_image_path = os.path.join(low_output_folder, f"{jpg_file[:-4]}.jpg")
    cv2.imwrite(output_image_path, low_pass_img)
    
    
    # Apply high-pss filter complement with the Gaussian blur
    high_pass_img = img - low_pass_img
    output_image_path = os.path.join(high_output_folder, f"{jpg_file[:-4]}.jpg")
    cv2.imwrite(output_image_path, high_pass_img)
    
#     # Create the sharpening kernel 
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    
#     # edge detection kernel (high pass)
# #     kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) 
    
#     sharpened_image = cv2.filter2D(img, -1, kernel) 
    
#     # Save the sharpened image to the output folder
#     output_image_path = os.path.join(sharpened_output_folder, f"{jpg_file[:-4]}.jpg")
#     cv2.imwrite(output_image_path, sharpened_image)

# reference: https://www.geeksforgeeks.org/implement-photoshop-high-pass-filter-hpf-using-opencv-in-python/

# try out which blurring size to use
jpg_file = jpg_files[0]
input_image_path = os.path.join(input_folder, jpg_file)
img = cv2.imread(input_image_path)

for blurring_size in np.arange(1, 101, 10):
    low_pass_img = cv2.GaussianBlur(img, (blurring_size, blurring_size), 0)

    # Save the low-pass filtered image to the output folder
    output_image_path = os.path.join("/Users/yizhen91/Document/UCR/3d_volumetric_video_streaming/gaussian-splatting/tandt/train/Gaussian_blur_level/", f"{jpg_file[:-4]}_{blurring_size}.jpg")
    cv2.imwrite(output_image_path, low_pass_img)