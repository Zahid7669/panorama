import cv2
import stitch as pano_stitch  
import utils as img_utils  
import os

# Set fixed paths for input and output
source_path = "data/room/"  # Update this to your source image directory
destination_path = "result/"    # Update this to your desired output directory
lower_resolution = 0  # Set to 1 to lower image resolution by 4x

# Load images from the specified directory
images_list = img_utils.fetchImages(source_path, lower_resolution)

# Stitch images to create a panorama
panorama_image = pano_stitch.stitchMultipleImages(images_list)

# Save the stitched image
output_file = os.path.join(destination_path, "panorama_result.jpg")
cv2.imwrite(output_file, panorama_image)


