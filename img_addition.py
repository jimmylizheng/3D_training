import cv2
import numpy as np

# Load the two images
# image1 = cv2.imread('images_high/00001.jpg')
# image2 = cv2.imread('images_low_31/00001.jpg')
image1 = cv2.imread('low_filtered_image.jpg')
image2 = cv2.imread('high_filtered_image.jpg')

# Check if the images have the same dimensions
if image1.shape == image2.shape:
    # Add the images pixel by pixel
    # result_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    result_image=image1+image2

    # Display or save the result image
    # cv2.imshow('Result Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the result image
    cv2.imwrite('result_image.jpg', result_image)
else:
    print("Images have different dimensions. They need to be of the same size for pixel-wise addition.")
