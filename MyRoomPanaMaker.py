import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


def get_distance_transform(img_rgb):
    thresh = cv.threshold(img_rgb, 0, 255, cv.THRESH_BINARY)[1]
    thresh = thresh.any(axis=2)
    thresh = np.pad(thresh, 1)
    dist = distance_transform_edt(thresh)[1:-1, 1:-1]
    dist = dist[:, :, None]
    max_dist = dist.max()
    if max_dist == 0:
        return np.zeros_like(dist, dtype=np.float64)
    return dist / max_dist * 255.0


def _preprocess_image(img, size):
    img_color = cv.resize(img, size)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    return img_gray, img_color


def _match_descriptors(desc1, desc2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def make_panorama(img1, img2, target_height=480, ransac_thresh=5.0,
                  show_matches=False, pair_name="pair"):
    h1, w1 = img1.shape[:2]
    scale1 = target_height / h1
    size1 = (int(w1 * scale1), int(h1 * scale1))
    h2, w2 = img2.shape[:2]
    scale2 = target_height / h2
    size2 = (int(w2 * scale2), int(h2 * scale2))
    gray1, color1 = _preprocess_image(img1, size1)
    gray2, color2 = _preprocess_image(img2, size2)

    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    matches = _match_descriptors(desc1, desc2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)

    if show_matches:
        n_show = min(50, len(matches))
        top_matches = matches[:n_show]
        vis_top = cv.drawMatches(
            color1, kp1, color2, kp2, top_matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        inlier_matches = []
        if mask is not None:
            inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
        vis_inliers = cv.drawMatches(
            color1, kp1, color2, kp2, inlier_matches[:n_show], None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        n_inliers = 0 if mask is None else int(mask.sum())
        print(f"[{pair_name}] keypoints: {len(kp1)} / {len(kp2)} | "
              f"matches: {len(matches)} | RANSAC inliers: {n_inliers}")

        cv.imshow(f"{pair_name}: top {n_show} matches", vis_top)
        cv.imshow(f"{pair_name}: RANSAC inliers (up to {n_show})", vis_inliers)
        cv.waitKey(0)
        cv.destroyWindow(f"{pair_name}: top {n_show} matches")
        cv.destroyWindow(f"{pair_name}: RANSAC inliers (up to {n_show})")

    h1, w1 = color1.shape[:2]
    h2, w2 = color2.shape[:2]

    output_width = int((w1 + w2) * 1)
    output_height = max(h1, h2)

    warped2 = cv.warpPerspective(color2, H, (output_width, output_height))

    canvas1 = np.zeros((output_height, output_width, 3), dtype=color1.dtype)
    canvas1[:h1, :w1] = color1

    weight1 = get_distance_transform(canvas1)
    weight2 = get_distance_transform(warped2)

    img1_f = canvas1.astype(np.float32)
    img2_f = warped2.astype(np.float32)

    denom = weight1 + weight2
    blended = (img1_f * weight1 + img2_f * weight2) / np.maximum(denom, 1.0)
    return blended.astype(np.uint8)


if __name__ == "__main__":
    left = cv.imread("images/My_room/IMG_8724.jpeg")
    right = cv.imread("images/My_room/IMG_8725.jpeg")
    panorama = make_panorama(left, right, show_matches=False, pair_name="0-1")

    left2 = cv.imread("images/My_room/IMG_8726.jpeg")
    right2 = cv.imread("images/My_room/IMG_8727.jpeg")
    panorama2 = make_panorama(left2, right2, show_matches=False, pair_name="2-3")



    megarama1 = make_panorama(panorama, panorama2, show_matches=True, pair_name="mega")
    

    cv.imshow("Panorama", panorama)
    cv.imshow("Panorama2", panorama2)

    cv.imshow("Megarama1", megarama1)


    cv.waitKey(0)
    cv.destroyAllWindows()
