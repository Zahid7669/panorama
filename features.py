import cv2
import numpy as np

def detect_and_describe(image, method="ORB"):
    """Detect and describe features in an image.
    Uses SURF, SIFT, or ORB algorithms based on the 'method' parameter.
    Returns the keypoints and descriptors of the image."""
    # Check if image depth is not CV_8U, then convert it to CV_8U
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Choose the feature detection method
    if method == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
    elif method == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    else:  # Default to ORB
        detector = cv2.ORB_create(nfeatures=3000)

    # Detect keypoints and compute descriptors
    kp, descriptors = detector.detectAndCompute(gray_img, None)
    descriptors = np.float32(descriptors)
    return kp, descriptors

def find_matches(descriptors1, descriptors2, ratio=0.75, method="FB"):
    """Match features between two sets of descriptors.
    Uses FlannBased or BruteForce matching based on the 'method' parameter.
    Applies Lowe's ratio test to filter matches. Returns the good matches."""
    if method == "BF":
        matcher = cv2.DescriptorMatcher_create("BruteForce")
    else:  # Default to FlannBased
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the top 2 matches for each descriptor
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    if len(good_matches) <= 4:
        raise Exception("Not enough matches")
    return good_matches

def compute_homography(source_img, destination_img, ransac_thresh=5.0):
    """Compute the homography matrix between two images.
    Detects and matches features, then uses RANSAC to compute the homography."""
    src_kp, src_desc = detect_and_describe(source_img)
    dst_kp, dst_desc = detect_and_describe(destination_img)

    matches = find_matches(src_desc, dst_desc)

    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, status.ravel().tolist()




