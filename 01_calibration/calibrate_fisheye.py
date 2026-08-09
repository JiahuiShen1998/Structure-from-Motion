"""Estimate fisheye intrinsics from chessboard stills.

Stage 01 of the pipeline. Detects the inner corners of a planar chessboard in a set of
calibration images, refines them to sub-pixel accuracy, and fits the Kannala-Brandt
fisheye model with `cv2.fisheye.calibrate`.

Input   a directory of stills of the same chessboard, all at the same resolution
Output  K (3x3 intrinsics), D (4 distortion coefficients), the RMS residual, one
        annotated image per successful detection, and a .npz that stage 02 consumes

The fisheye model is used rather than `cv2.calibrateCamera` because the lens covers a
wide enough field that the pinhole polynomial radial model does not fit the periphery.
The model is theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8),
acting on the incidence angle rather than on pixel radius -- which is why D is
resolution-independent while K is not.

Expect attrition: on a fisheye lens the board bows near the frame border and
findChessboardCorners fails outright on a substantial fraction of the set. See the
stage README.

The algorithm and every numeric parameter are unchanged from the original
`calibration_1.py`; this file adds a CLI, docstrings and result persistence.
"""

import argparse
import glob
import os

import cv2
import numpy as np

# Inner corner count, i.e. one less than the square count in each direction.
# A 10x8-square board gives (9, 7). Getting this wrong makes detection fail on
# every image with no other symptom.
CHECKERBOARD = (9, 7)

# Sub-pixel corner refinement: stop after 300 iterations or when the corner moves
# less than 0.01 px.
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.01)
SUBPIX_WINDOW = (13, 13)

# Calibration solver: 300 iterations or 1e-6 change.
CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)

DETECT_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE)


def build_object_points(pattern):
    """Build the 3D coordinates of the board's inner corners, in board frame.

    The board is planar, so z = 0 throughout. Coordinates are in units of *chessboard
    squares*, not millimetres -- the original code left the `objp *= square_size` line
    commented out. K and D are unaffected by this (they depend only on the ratio of
    object to image coordinates), but any translation vector recovered later is on an
    arbitrary scale.
    """
    objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    return objp


def detect_corners(images, pattern, out_dir=None, show=False):
    """Detect and sub-pixel-refine chessboard corners across a set of images.

    Args:
        images: list of image paths, all the same resolution.
        pattern: (cols, rows) of inner corners.
        out_dir: if given, write one annotated image per successful detection.
        show: display each detection in a window for 500 ms.

    Returns:
        (objpoints, imgpoints, image_size, accepted, rejected) where `image_size` is
        (width, height) and `rejected` lists the paths detection failed on.

    Raises:
        ValueError: if the inputs are not all the same resolution. Calibration is only
            valid for a single resolution; mixing them silently produces a meaningless
            K.
    """
    objp = build_object_points(pattern)
    objpoints, imgpoints = [], []
    accepted, rejected = [], []
    img_shape = None

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] could not read {fname}, skipping")
            continue

        if img_shape is None:
            img_shape = img.shape[:2]
        elif img_shape != img.shape[:2]:
            raise ValueError(
                f"{fname} is {img.shape[1]}x{img.shape[0]}, expected "
                f"{img_shape[1]}x{img_shape[0]}. All calibration images must share "
                f"one resolution."
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, DETECT_FLAGS)

        if not found:
            rejected.append(fname)
            continue

        # Refines `corners` in place; the window is the search radius. Too large on a
        # fisheye frame and it locks onto a curved neighbouring edge.
        cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA)

        objpoints.append(objp)
        imgpoints.append(corners)
        accepted.append(fname)

        cv2.drawChessboardCorners(img, pattern, corners, found)
        if show:
            cv2.imshow("Corners", img)
            cv2.waitKey(500)
        if out_dir:
            stem = os.path.splitext(os.path.basename(fname))[0]
            cv2.imwrite(os.path.join(out_dir, f"corners_{stem}.png"), img)

    if show:
        cv2.destroyAllWindows()

    if img_shape is None:
        raise ValueError("No readable images found.")

    return objpoints, imgpoints, (img_shape[1], img_shape[0]), accepted, rejected


def calibrate(objpoints, imgpoints, image_size, check_cond=False):
    """Fit the fisheye model to the detected corners.

    Args:
        objpoints, imgpoints: as returned by `detect_corners`.
        image_size: (width, height).
        check_cond: enable `CALIB_CHECK_COND`. With it on, OpenCV raises as soon as a
            single view makes the system ill-conditioned and names that view -- useful
            when the residual looks acceptable but rectification does not. With it off
            (the original behaviour, and the default here) the fit proceeds and absorbs
            that view's error into K and D silently.

    Returns:
        (rms, K, D). D holds k1..k4 and acts on the incidence angle, so unlike K it
        does not scale with resolution.
    """
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    if check_cond:
        flags += cv2.fisheye.CALIB_CHECK_COND

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_size, K, D, None, None, flags, CALIB_CRITERIA
    )
    return rms, K, D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate fisheye intrinsics from chessboard stills (stage 01)."
    )
    parser.add_argument("--images", required=True,
                        help="Directory of calibration images, or a glob pattern")
    parser.add_argument("--out", default="calib.npz",
                        help="Where to write K, D and DIM (default: calib.npz)")
    parser.add_argument("--pattern", default="9x7",
                        help="Inner corner count as COLSxROWS, one less than the "
                             "square count each way (default: 9x7)")
    parser.add_argument("--corners-dir", default=None,
                        help="Directory for annotated detections "
                             "(default: outputs/corners next to --out)")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated filenames to leave out, e.g. "
                             "calibrate19_RIGHT.png. Use this to drop a view that "
                             "makes the solver fail; see the stage README")
    parser.add_argument("--check-cond", action="store_true",
                        help="Enable CALIB_CHECK_COND: abort and name the view that "
                             "makes the system ill-conditioned")
    parser.add_argument("--show", action="store_true",
                        help="Display each detection in a window")
    return parser.parse_args()


def main():
    args = parse_args()

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern = (cols, rows)

    if any(ch in args.images for ch in "*?["):
        images = sorted(glob.glob(args.images))
    else:
        images = sorted(glob.glob(os.path.join(args.images, "*")))
        images = [p for p in images
                  if os.path.splitext(p)[1].lower() in
                  (".png", ".jpg", ".jpeg", ".bmp", ".tiff")]
    if not images:
        raise SystemExit(f"No images matched {args.images}")

    excluded = {n.strip() for n in args.exclude.split(",") if n.strip()}
    if excluded:
        images = [p for p in images if os.path.basename(p) not in excluded]
        print(f"[INFO] excluded {len(excluded)} image(s) by request")

    corners_dir = args.corners_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "outputs", "corners")

    objpoints, imgpoints, image_size, accepted, rejected = detect_corners(
        images, pattern, out_dir=corners_dir, show=args.show)

    print(f"[INFO] detected corners in {len(accepted)}/{len(images)} images")
    if rejected:
        names = ", ".join(os.path.basename(p) for p in rejected)
        print(f"[INFO] no detection in: {names}")
        print("[INFO] attrition is normal on a fisheye lens; see the stage README")

    if not objpoints:
        raise SystemExit("No valid images for calibration.")

    try:
        rms, K, D = calibrate(objpoints, imgpoints, image_size,
                              check_cond=args.check_cond)
    except cv2.error as exc:
        raise SystemExit(
            f"cv2.fisheye.calibrate failed: "
            f"{str(exc).strip().splitlines()[-1]}\n"
            "This usually means one or two views are degenerate. Re-run with "
            "--check-cond to confirm, then bisect with --exclude to find them. "
            "See the stage README."
        ) from exc

    print(f"[INFO] RMS reprojection error: {rms:.4f} px")
    if rms > 10:
        print("[WARN] an RMS this large means the fit did not really converge; "
              "a healthy chessboard calibration lands around 1 px")
    print(f"DIM = {image_size}")
    print(f"K = np.array({K.tolist()})")
    print(f"D = np.array({D.tolist()})")

    np.savez(args.out, K=K, D=D, DIM=np.array(image_size), rms=rms,
             num_images=len(accepted))
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
