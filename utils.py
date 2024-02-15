from imutils import paths
import cv2
import numpy as np

def fetchImages(directory, downsize):
    """Fetch images from directory, resizing if specified. @param directory is the path containing images,
    @param downsize is True to reduce image size to a quarter, otherwise False."""
    paths_list = sorted(list(paths.list_images(directory)))
    images_collected = []
    for path in paths_list:
        img = cv2.imread(path)
        if downsize:
            img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        images_collected.append(img)
    return images_collected

