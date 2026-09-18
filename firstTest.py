import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt

def get_distance_transform(img_rgb):
    """
    Get distance to the closest background pixel for an RGB image
    Input:
        img_rgb: np.array, HxWx3 RGB image
    Output:
        dist: np.array, HxWx1 distance image
        each pixel's intensity is proportional to
        its distance to the closest background pixel
        scaled to [0..255] for plotting
    """
    # Threshold the image: any value above 0 maps into 255
    thresh = cv.threshold(img_rgb, 0, 255, cv.THRESH_BINARY)[1]
    # Collapse the color dimension
    thresh = thresh.any(axis=2)
    # Pad to make sure the border is treated as background
    thresh = np.pad(thresh, 1)
    # Get distance transform
    dist = distance_transform_edt(thresh)[1:-1, 1:-1]
    # HxW -> HxWx1
    dist = dist[:, :, None]
    return dist / dist.max() * 255.0


def preprocess_image(img, size):
    """Resize to size and return (gray, color) versions."""
    img_color = cv.resize(img, size)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    return img_gray, img_color


def match_descriptors(desc1, desc2):
    """Match descriptors with BFMatcher and sort by distance."""
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def main():
    img1 = cv.imread("images/yosemite/yosemite1.jpg")
    img2 = cv.imread("images/yosemite/yosemite2.jpg")
    img3 = cv.imread("images/yosemite/yosemite3.jpg")
    img4 = cv.imread("images/yosemite/yosemite4.jpg")

    h, w = img1.shape[:2]
    scale = 480 / h
    size = (int(w * scale), int(h * scale))

    img1, img1_color = preprocess_image(img1, size)
    img2, img2_color = preprocess_image(img2, size)
    img3, img3_color = preprocess_image(img3, size)
    img4, img4_color = preprocess_image(img4, size)

    sift = cv.SIFT_create()
    sift_keypoints1, sift_descriptors1 = sift.detectAndCompute(img1, None)
    sift_keypoints2, sift_descriptors2 = sift.detectAndCompute(img2, None)
    sift_keypoints3, sift_descriptors3 = sift.detectAndCompute(img3, None)
    sift_keypoints4, sift_descriptors4 = sift.detectAndCompute(img4, None)

    matches1 = match_descriptors(sift_descriptors1, sift_descriptors2)
    matches2 = match_descriptors(sift_descriptors3, sift_descriptors4)

    pts11 = np.float32([sift_keypoints1[m.queryIdx].pt for m in matches1])
    pts12 = np.float32([sift_keypoints2[m.trainIdx].pt for m in matches1])

    pts21 = np.float32([sift_keypoints3[m.queryIdx].pt for m in matches2])
    pts22 = np.float32([sift_keypoints4[m.trainIdx].pt for m in matches2])

    H12, mask12 = cv.findHomography(pts12, pts11, cv.RANSAC, 5.0)
    H34, mask34 = cv.findHomography(pts21, pts22, cv.RANSAC, 5.0)

    h,w = img1.shape[:2]
    warped_2_to_1 = cv.warpPerspective(img2_color, H12, (w, h))

    h, w = img3.shape[:2]
    warped_3_to_4 = cv.warpPerspective(img3_color, H34, (w, h))

    weight1 = get_distance_transform(img1_color)
    weight2 = get_distance_transform(warped_2_to_1)

    weight3 = get_distance_transform(img3_color)
    weight4 = get_distance_transform(warped_3_to_4)

    img1_f = img1_color.astype(np.float32)
    img2_f = warped_2_to_1.astype(np.float32)

    img4_f = img4_color.astype(np.float32)
    img3_f = warped_3_to_4.astype(np.float32)

    denom = weight1 + weight2
    blended12 = (img1_f * weight1 + img2_f * weight2) / np.maximum(denom, 1.0)
    blended12 = blended12.astype("uint8")

    denom = weight3 + weight4
    blended34 = (img4_f * weight3 + img3_f * weight4) / np.maximum(denom, 1.0)
    blended34 = blended34.astype("uint8")

    cv.imshow("Blend 1+2", blended12)
    cv.imshow("Blend 3+4", blended34)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
