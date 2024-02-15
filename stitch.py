import cv2
import numpy as np
import features

def createBlendingMask(img_height, img_width, transition_point, blend_width, favor_left=True):
    assert transition_point < img_width
    blend_mask = np.zeros((img_height, img_width))

    offset = int(blend_width / 2)
    try:
        if favor_left:
            blend_mask[:, transition_point - offset : transition_point + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (img_height, 1)
            )
            blend_mask[:, : transition_point - offset] = 1
        else:
            blend_mask[:, transition_point - offset : transition_point + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (img_height, 1)
            )
            blend_mask[:, transition_point + offset + 1 :] = 1
    except Exception:
        if favor_left:
            blend_mask[:, transition_point - offset : transition_point + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (img_height, 1)
            )
            blend_mask[:, : transition_point - offset] = 1
        else:
            blend_mask[:, transition_point - offset : transition_point + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (img_height, 1)
            )
            blend_mask[:, transition_point + offset + 1 :] = 1

    return cv2.merge([blend_mask, blend_mask, blend_mask])

def blendPanoramas(pano_base_resized, pano_to_blend, base_width, merge_side, display_steps=False):
    pano_height, pano_width, _ = pano_base_resized.shape
    blend_region = int(base_width / 8)
    transition = base_width - int(blend_region / 2)
    mask_base = createBlendingMask(pano_height, pano_width, transition, blend_width=blend_region, favor_left=True)
    mask_blend = createBlendingMask(pano_height, pano_width, transition, blend_width=blend_region, favor_left=False)

    if display_steps:
        combined = pano_to_blend + pano_base_resized
    else:
        combined = None
        left_part = None
        right_part = None

    if merge_side == "left":
        pano_base_resized = cv2.flip(pano_base_resized, 1)
        pano_to_blend = cv2.flip(pano_to_blend, 1)
        pano_base_resized = pano_base_resized * mask_base
        pano_to_blend = pano_to_blend * mask_blend
        merged_pano = pano_to_blend + pano_base_resized
        merged_pano = cv2.flip(merged_pano, 1)
        if display_steps:
            left_part = cv2.flip(pano_to_blend, 1)
            right_part = cv2.flip(pano_base_resized, 1)
    else:
        pano_base_resized = pano_base_resized * mask_base
        pano_to_blend = pano_to_blend * mask_blend
        merged_pano = pano_to_blend + pano_base_resized
        if display_steps:
            left_part = pano_base_resized
            right_part = pano_to_blend

    return merged_pano, combined, left_part, right_part

def mergeImages(src_image, dst_image, display_steps=False):
    homography_matrix, _ = features.compute_homography(src_image, dst_image)

    src_height, src_width = src_image.shape[:2]
    dst_height, dst_width = dst_image.shape[:2]

    corners_src = np.float32([[0, 0], [0, src_height], [src_width, src_height], [src_width, 0]]).reshape(-1, 1, 2)
    corners_dst = np.float32([[0, 0], [0, dst_height], [dst_width, dst_height], [dst_width, 0]]).reshape(-1, 1, 2)

    try:
        corners_src_transformed = cv2.perspectiveTransform(corners_src, homography_matrix)
        all_corners = np.concatenate((corners_src_transformed, corners_dst), axis=0)

        [x_min, y_min] = np.int64(all_corners.min(axis=0).ravel() - 0.5)
        [_, y_max] = np.int64(all_corners.max(axis=0).ravel() + 0.5)
        translation = [-x_min, -y_min]

        if corners_src_transformed[0][0][0] < 0:
            merge_direction = "left"
            pano_width = dst_width + translation[0]
        else:
            merge_direction = "right"
            pano_width = int(corners_src_transformed[3][0][0])
        pano_height = y_max - y_min

        translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        warped_src_image = cv2.warpPerspective(src_image, translation_matrix.dot(homography_matrix), (pano_width, pano_height))

        resized_dst_image = np.zeros((pano_height, pano_width, 3))
        if merge_direction == "left":
            resized_dst_image[translation[1] : src_height + translation[1], translation[0] : dst_width + translation[0]] = dst_image
        else:
            resized_dst_image[translation[1] : src_height + translation[1], :dst_width] = dst_image

        blended_panorama, _, _, _ = blendPanoramas(resized_dst_image, warped_src_image, dst_width, merge_direction, display_steps=display_steps)
        cropped_panorama = trimPanorama(blended_panorama, dst_height, all_corners)
        return cropped_panorama, _, _, _
    except Exception:
        raise Exception("Error processing images. Please try another set of images.")

def stitchMultipleImages(images_list):
    mid_index = int(len(images_list) / 2 + 0.5)
    left_group = images_list[:mid_index]
    right_group = images_list[mid_index - 1:]
    right_group.reverse()

    while len(left_group) > 1:
        base_image = left_group.pop()
        merge_image = left_group.pop()
        left_merged, _, _, _ = mergeImages(merge_image, base_image)
        left_group.append(left_merged.astype("uint8"))

    while len(right_group) > 1:
        base_image = right_group.pop()
        merge_image = right_group.pop()
        right_merged, _, _, _ = mergeImages(merge_image, base_image)
        right_group.append(right_merged.astype("uint8"))

    if right_merged.shape[1] >= left_merged.shape[1]:
        final_panorama, _, _, _ = mergeImages(left_merged, right_merged)
    else:
        final_panorama, _, _, _ = mergeImages(right_merged, left_merged)
    return final_panorama

def trimPanorama(panorama_img, dst_height, corners):
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    translation = [-x_min, -y_min]
    corners = corners.astype(int)

    if corners[0][0][0] < 0:
        offset = abs(corners[1][0][0] - corners[0][0][0])
        trimmed_panorama = panorama_img[translation[1] : dst_height + translation[1], offset:, :]
    else:
        max_x_corner = min(corners[2][0][0], corners[3][0][0])
        trimmed_panorama = panorama_img[translation[1] : dst_height + translation[1], :max_x_corner, :]
    return trimmed_panorama
