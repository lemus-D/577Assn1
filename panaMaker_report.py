"""
Copy of panaMaker.py with cv.imshow at HW1 Steps 1–4 for report screenshots.
Press any key in each window to advance.
"""
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


def _show(title, img):
    """Show an image and wait for a key so you can screenshot."""
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyWindow(title)


def make_panorama(img1, img2, target_height=480, ransac_thresh=5.0,
                  show_steps=False, pair_name="pair"):
    h1, w1 = img1.shape[:2]
    scale1 = target_height / h1
    size1 = (int(w1 * scale1), int(h1 * scale1))
    h2, w2 = img2.shape[:2]
    scale2 = target_height / h2
    size2 = (int(w2 * scale2), int(h2 * scale2))
    gray1, color1 = _preprocess_image(img1, size1)
    gray2, color2 = _preprocess_image(img2, size2)

    # --- Step 1: Feature detection and description ---
    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if show_steps:
        kp_vis1 = cv.drawKeypoints(
            color1, kp1, None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        kp_vis2 = cv.drawKeypoints(
            color2, kp2, None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        print(f"[{pair_name} Step 1] keypoints: {len(kp1)} / {len(kp2)}")
        _show(f"{pair_name} Step1: SIFT keypoints (left)", kp_vis1)
        _show(f"{pair_name} Step1: SIFT keypoints (right)", kp_vis2)

    # --- Step 2: Feature matching and homography / warp ---
    matches = _match_descriptors(desc1, desc2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)

    h1, w1 = color1.shape[:2]
    h2, w2 = color2.shape[:2]
    output_width = w1 + w2
    output_height = max(h1, h2)

    warped2 = cv.warpPerspective(color2, H, (output_width, output_height))

    canvas1 = np.zeros((output_height, output_width, 3), dtype=color1.dtype)
    canvas1[:h1, :w1] = color1

    if show_steps:
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
        print(f"[{pair_name} Step 2] matches: {len(matches)} | "
              f"RANSAC inliers: {n_inliers}")

        _show(f"{pair_name} Step2: top {n_show} matches", vis_top)
        _show(f"{pair_name} Step2: RANSAC inliers (up to {n_show})", vis_inliers)
        _show(f"{pair_name} Step2: warped right image", warped2)

        # Intensity sum (HW Fig. 3 style alignment check)
        sum_vis = np.clip(
            canvas1.astype(np.float32) + warped2.astype(np.float32), 0, 255
        ).astype(np.uint8)
        _show(f"{pair_name} Step2: intensity sum (alignment check)", sum_vis)

    # --- Step 3: Image blending with distance-transform weights ---
    weight1 = get_distance_transform(canvas1)
    weight2 = get_distance_transform(warped2)

    img1_f = canvas1.astype(np.float32)
    img2_f = warped2.astype(np.float32)

    denom = weight1 + weight2
    blended = (img1_f * weight1 + img2_f * weight2) / np.maximum(denom, 1.0)
    blended = blended.astype(np.uint8)

    if show_steps:
        w1_vis = np.clip(weight1, 0, 255).astype(np.uint8)
        w2_vis = np.clip(weight2, 0, 255).astype(np.uint8)
        print(f"[{pair_name} Step 3] blended panorama shown")
        _show(f"{pair_name} Step3: distance weight (left)", w1_vis)
        _show(f"{pair_name} Step3: distance weight (warped right)", w2_vis)
        _show(f"{pair_name} Step3: blended pair", blended)

    return blended


if __name__ == "__main__":
    left = cv.imread("images/yosemite/yosemite1.jpg")
    right = cv.imread("images/yosemite/yosemite2.jpg")
    # Full Step 1–3 visualization on the first pair
    panorama = make_panorama(
        left, right, show_steps=True, pair_name="1-2",
    )

    left2 = cv.imread("images/yosemite/yosemite3.jpg")
    right2 = cv.imread("images/yosemite/yosemite4.jpg")
    # Second pair without intermediate windows (still needed for Step 4)
    panorama2 = make_panorama(
        left2, right2, show_steps=False, pair_name="3-4",
    )

    megarama = make_panorama(
        panorama, panorama2, show_steps=False, pair_name="mega",
    )

    # --- Step 4: Stitching multiple images (progressive pairwise) ---
    print("[Step 4] progressive stitch results")
    _show("Step4: panorama (images 1+2)", panorama)
    _show("Step4: panorama2 (images 3+4)", panorama2)
    _show("Step4: megarama (all four)", megarama)

    cv.destroyAllWindows()
