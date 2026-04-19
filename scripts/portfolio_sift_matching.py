import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np


def detect_shi_corners(img, max_corners=2000, quality_level=0.01, min_distance=10):
    """
    Detect Shi-Tomasi corners and convert them to OpenCV KeyPoint objects.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if corners is None:
        return []

    keypoints = [
        cv.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=50)
        for pt in corners
    ]
    return keypoints


def load_images(img1_path, img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Could not read image 1: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read image 2: {img2_path}")

    return img1, img2


def build_matcher(use_flann=True):
    if use_flann:
        index_params = dict(algorithm=1, trees=4)
        search_params = dict(checks=50)
        return cv.FlannBasedMatcher(index_params, search_params)

    return cv.BFMatcher(cv.NORM_L2)


def ratio_test_filter(matches, ratio=0.7):
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def compute_fundamental_ransac(keypoints1, keypoints2, matches, ransac_thresh=5.0):
    if len(matches) < 8:
        return None, [], None

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, ransac_thresh)

    if F is None or mask is None:
        return None, [], None

    matches_mask = mask.ravel().tolist()
    ransac_matches = [matches[i] for i in range(len(matches)) if matches_mask[i]]

    return F, ransac_matches, matches_mask


def save_keypoint_visualization(img, keypoints, out_path):
    color = (0, 250, 0)
    vis = cv.drawKeypoints(img, keypoints, None, color=color, flags=2)
    cv.imwrite(out_path, vis)


def ensure_output_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def run_feature_matching(
    img1_path,
    img2_path,
    output_dir="outputs",
    num_matches=250,
    use_flann=True,
    max_corners=2000,
    quality_level=0.01,
    min_distance=10,
    ratio_threshold=0.7,
    ransac_thresh=5.0,
):
    """
    Clean showcase version adapted from the original project code.

    Pipeline:
    1. Load two images
    2. Detect Shi-Tomasi corners
    3. Compute SIFT descriptors at those corner locations
    4. Match descriptors using FLANN or BFMatcher
    5. Apply Lowe's ratio test
    6. Estimate the fundamental matrix with RANSAC
    7. Save visualizations
    """
    ensure_output_dir(output_dir)

    img1, img2 = load_images(img1_path, img2_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    corners1 = detect_shi_corners(
        img1,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
    )
    corners2 = detect_shi_corners(
        img2,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
    )

    if len(corners1) == 0 or len(corners2) == 0:
        raise RuntimeError("No Shi-Tomasi corners detected in one or both images.")

    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.compute(gray1, corners1)
    keypoints2, descriptors2 = sift.compute(gray2, corners2)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("SIFT descriptor computation failed.")

    print(f"[INFO] Keypoints in image 1: {len(keypoints1)}")
    print(f"[INFO] Keypoints in image 2: {len(keypoints2)}")

    save_keypoint_visualization(
        img1,
        keypoints1,
        os.path.join(output_dir, "keypoints_img1.jpg")
    )
    save_keypoint_visualization(
        img2,
        keypoints2,
        os.path.join(output_dir, "keypoints_img2.jpg")
    )

    matcher = build_matcher(use_flann=use_flann)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = ratio_test_filter(matches, ratio=ratio_threshold)
    good_matches = good_matches[:num_matches]

    print(f"[INFO] Matches after ratio test: {len(good_matches)}")

    F, ransac_matches, matches_mask = compute_fundamental_ransac(
        keypoints1,
        keypoints2,
        good_matches,
        ransac_thresh=ransac_thresh,
    )

    if F is None:
        print("[WARNING] Fundamental matrix estimation failed or not enough matches.")
        ransac_matches = []
    else:
        print("[INFO] Fundamental matrix:")
        print(F)
        print(f"[INFO] Inlier matches after RANSAC: {len(ransac_matches)}")

    green = (0, 250, 0)

    matched_image = cv.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        good_matches,
        None,
        matchColor=green,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv.imwrite(
        os.path.join(output_dir, "matches_before_ransac.jpg"),
        matched_image
    )

    if len(ransac_matches) > 0:
        ransac_matches_image = cv.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            ransac_matches[:num_matches],
            None,
            matchColor=green,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.imwrite(
            os.path.join(output_dir, "matches_after_ransac.jpg"),
            ransac_matches_image
        )

    print(f"[INFO] Results saved in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio showcase: Shi-Tomasi + SIFT + RANSAC feature matching demo"
    )
    parser.add_argument("--img1", required=True, help="Path to the first image")
    parser.add_argument("--img2", required=True, help="Path to the second image")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num_matches",
        type=int,
        default=250,
        help="Maximum number of matches to visualize"
    )
    parser.add_argument(
        "--use_flann",
        action="store_true",
        help="Use FLANN matcher instead of BFMatcher"
    )
    parser.add_argument(
        "--max_corners",
        type=int,
        default=2000,
        help="Maximum number of Shi-Tomasi corners"
    )
    parser.add_argument(
        "--quality_level",
        type=float,
        default=0.01,
        help="Shi-Tomasi quality level"
    )
    parser.add_argument(
        "--min_distance",
        type=int,
        default=10,
        help="Minimum distance between Shi-Tomasi corners"
    )
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=0.7,
        help="Lowe ratio threshold"
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=5.0,
        help="RANSAC threshold for fundamental matrix estimation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_feature_matching(
        img1_path=args.img1,
        img2_path=args.img2,
        output_dir=args.output_dir,
        num_matches=args.num_matches,
        use_flann=args.use_flann,
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        ratio_threshold=args.ratio_threshold,
        ransac_thresh=args.ransac_thresh,
    )
