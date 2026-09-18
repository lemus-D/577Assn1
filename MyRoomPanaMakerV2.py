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


def make_panorama(merged, mergeTo, merge_side="left", target_height=480,
                  ransac_thresh=5.0, show_matches=False, pair_name="pair"):

    h_to, w_to = mergeTo.shape[:2]
    scale_to = target_height / h_to
    size_to = (int(w_to * scale_to), int(h_to * scale_to))
    h_m, w_m = merged.shape[:2]
    scale_m = target_height / h_m
    size_m = (int(w_m * scale_m), int(h_m * scale_m))

    gray_to, color_to = _preprocess_image(mergeTo, size_to)
    gray_m, color_m = _preprocess_image(merged, size_m)

    sift = cv.SIFT_create()
    kp_to, desc_to = sift.detectAndCompute(gray_to, None)
    kp_m, desc_m = sift.detectAndCompute(gray_m, None)

    # query = mergeTo, train = merged  →  Homography maps merged → mergeTo
    matches = _match_descriptors(desc_to, desc_m)

    pts_to = np.float32([kp_to[m.queryIdx].pt for m in matches])
    pts_m = np.float32([kp_m[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts_m, pts_to, cv.RANSAC, ransac_thresh)

    if show_matches:
        n_show = min(50, len(matches))
        top_matches = matches[:n_show]
        vis_top = cv.drawMatches(
            color_to, kp_to, color_m, kp_m, top_matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        inlier_matches = []
        if mask is not None:
            inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
        vis_inliers = cv.drawMatches(
            color_to, kp_to, color_m, kp_m, inlier_matches[:n_show], None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        n_inliers = 0 if mask is None else int(mask.sum())
        print(f"[{pair_name}] keypoints: {len(kp_to)} / {len(kp_m)} | "
              f"matches: {len(matches)} | RANSAC inliers: {n_inliers}")

        cv.imshow(f"{pair_name}: top {n_show} matches", vis_top)
        cv.imshow(f"{pair_name}: RANSAC inliers (up to {n_show})", vis_inliers)
        cv.waitKey(0)
        cv.destroyWindow(f"{pair_name}: top {n_show} matches")
        cv.destroyWindow(f"{pair_name}: RANSAC inliers (up to {n_show})")

    h_to, w_to = color_to.shape[:2]
    h_m, w_m = color_m.shape[:2]

    output_width = w_to + w_m
    output_height = max(h_to, h_m)

    # Shift so warped content that lands left/above mergeTo still fits on the canvas.
    corners_m = np.float32([[0, 0], [w_m, 0], [w_m, h_m], [0, h_m]]).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners_m, H).reshape(-1, 2)
    min_x = float(warped_corners[:, 0].min())
    min_y = float(warped_corners[:, 1].min())
    max_x = float(warped_corners[:, 0].max())
    max_y = float(warped_corners[:, 1].max())

    ty = max(0.0, -min_y)
    if merge_side == "left":
        # mergeTo stays near the left; only translate if the warp goes negative.
        tx = max(0.0, -min_x)
    else:
        # Place mergeTo on the right: leave room on the left for the warped image.
        tx = max(-min_x, float(w_m))
    merge_to_x = int(round(tx))

    output_width = int(max(output_width, np.ceil(max_x + tx), merge_to_x + w_to))
    output_height = int(max(output_height, np.ceil(max_y + ty), ty + h_to))

    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    H_adj = T @ H

    warped_m = cv.warpPerspective(color_m, H_adj, (output_width, output_height))

    canvas_to = np.zeros((output_height, output_width, 3), dtype=color_to.dtype)
    y0 = int(round(ty))
    canvas_to[y0:y0 + h_to, merge_to_x:merge_to_x + w_to] = color_to

    weight1 = get_distance_transform(canvas_to)
    weight2 = get_distance_transform(warped_m)

    img1_f = canvas_to.astype(np.float32)
    img2_f = warped_m.astype(np.float32)

    denom = weight1 + weight2
    blended = (img1_f * weight1 + img2_f * weight2) / np.maximum(denom, 1.0)
    return blended.astype(np.uint8)


if __name__ == "__main__":
    # Center anchors: IMG_2 and IMG_3. Outer images warp into them.
    # Sequence left→right: IMG_1, IMG_2, IMG_3, IMG_4
    merged1 = cv.imread("images/My_room/IMG_1.jpeg")   # left of center
    mergeTo1 = cv.imread("images/My_room/IMG_2.jpeg")  # center-left anchor
    # Warp IMG_1 into IMG_2; mergeTo sits on the right of this half.
    panorama = make_panorama(
        merged1, mergeTo1, merge_side="right",
        show_matches=False, pair_name="1→2",
    )

    mergeTo2 = cv.imread("images/My_room/IMG_3.jpeg")  # center-right anchor
    merged2 = cv.imread("images/My_room/IMG_4.jpeg")   # right of center
    # Warp IMG_4 into IMG_3; mergeTo sits on the left of this half.
    panorama2 = make_panorama(
        merged2, mergeTo2, merge_side="left",
        show_matches=False, pair_name="4→3",
    )

    # Merge left half into right half (toward the IMG_2 / IMG_3 center).
    megarama1 = make_panorama(
        panorama, panorama2, merge_side="right",
        show_matches=True, pair_name="mega",
    )

    cv.imshow("Panorama", panorama)
    cv.imshow("Panorama2", panorama2)
    cv.imshow("Megarama1", megarama1)

    cv.waitKey(0)
    cv.destroyAllWindows()
