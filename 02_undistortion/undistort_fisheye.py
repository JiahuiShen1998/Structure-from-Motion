"""Rectify fisheye frames to a pinhole projection.

Stage 02 of the pipeline. Takes the intrinsics estimated in stage 01 and maps every
input frame onto a pinhole camera, so that straight lines in the scene stay straight
and the SIFT descriptors used downstream see locally affine neighbourhoods.

Input   a directory of fisheye frames, plus K, D and the resolution K was estimated at
Output  one rectified image per input, written to --out

The frames may be at a different resolution than the calibration stills, which is the
usual case here: calibration was done on 1920x1080 chessboard shots, the video frames
are 3840x2160. K is scaled accordingly; D is not, because the fisheye coefficients act
on the normalised incidence angle and carry no pixel units.

Provenance
----------
No undistortion script was ever committed to the original team repository, although its
outputs were. The `undistort()` body below is the listing from the project report
(`thesis/texfiles/implementation.tex`), and the default K, D and DIM are the values
from the team's working copy (`undistort.py`). Only the CLI, the batch loop and the
docstrings are new; the algorithm is unchanged.

A second, simpler variant also existed in the working copy: it called
`initUndistortRectifyMap(K, D, eye(3), K, DIM, ...)` directly, using K itself as the
output camera matrix and running only at the calibration resolution. That variant
cannot handle the 4K frames, so the report version is the one kept here. Pass
--new-camera same-as-k to reproduce it.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Calibration produced by 01_calibration/calibrate_fisheye.py, valid at DIM.
# Three calibration runs were recorded in the team's working copy with residuals
# 204.5, 227.8 and 302.0; the lowest-residual one is used here. See the stage README.
DIM = (1920, 1080)
K = np.array([
    [591.0412707624, 0.0, 929.1522869318215],
    [0.0, 594.439320988173, 535.87156962935],
    [0.0, 0.0, 1.0],
])
D = np.array([
    [-0.005214816763725941],
    [-0.030490227834563968],
    [0.009805068492885203],
    [-0.00013176708616799765],
])

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def build_undistort_maps(K, D, DIM, size, balance=0.0, fov_scale=0.8,
                         new_camera="estimated"):
    """Build the remap tables that rectify a fisheye image of the given size.

    The maps depend only on the intrinsics and the resolution, so build them once and
    reuse them across a sequence -- this is the expensive call, not `cv2.remap`.

    Args:
        K: 3x3 intrinsic matrix, valid at resolution `DIM`.
        D: 4x1 Kannala-Brandt coefficients (k1..k4). Acts on the incidence angle
            theta, not on pixel coordinates, so it is resolution-independent and must
            NOT be rescaled.
        DIM: (width, height) that `K` was estimated at.
        size: (width, height) of the images being rectified.
        balance: 0.0 crops the output to the largest all-valid rectangle -- no black
            wedges, narrowest field of view. 1.0 keeps the full field and leaves the
            corners invalid. Black wedges generate spurious SIFT keypoints along their
            edges that survive the ratio test, which is why 0.0 is the default.
        fov_scale: divides the output focal length, widening the field that fits in
            the frame at the cost of resolution per pixel. Below ~0.6 the periphery is
            stretched enough to disturb SIFT's scale estimates.
        new_camera: "estimated" uses
            `fisheye.estimateNewCameraMatrixForUndistortRectify`; "same-as-k" uses the
            scaled K itself, reproducing the simpler variant described in the module
            docstring (balance and fov_scale are then ignored).

    Returns:
        (map1, map2) in CV_16SC2 fixed-point form -- roughly 2x faster in `remap` than
        float maps, at sub-pixel accuracy that does not matter here.

    Raises:
        ValueError: if `size` has a different aspect ratio than `DIM`. A mismatch is
            otherwise silent: the output looks plausible but is geometrically wrong.
    """
    if abs(size[0] / size[1] - DIM[0] / DIM[1]) > 1e-6:
        raise ValueError(
            f"Image aspect ratio {size[0]}x{size[1]} does not match calibration "
            f"{DIM[0]}x{DIM[1]}. K is only valid for one aspect ratio."
        )

    # K carries pixel units, so it scales with resolution. The homogeneous 1 must not.
    scaled_K = K * size[0] / DIM[0]
    scaled_K[2][2] = 1.0

    if new_camera == "same-as-k":
        new_K = scaled_K
    else:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K, D, size, np.eye(3), balance=balance, fov_scale=fov_scale
        )

    return cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, size, cv2.CV_16SC2
    )


def undistort_directory(images_dir, out_dir, K, D, DIM, balance=0.0, fov_scale=0.8,
                        new_camera="estimated"):
    """Rectify every image in `images_dir` into `out_dir`, reusing one remap table."""
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    maps = None
    map_size = None
    written = 0

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] could not read {path.name}, skipping")
            continue

        size = (img.shape[1], img.shape[0])
        if maps is None or size != map_size:
            maps = build_undistort_maps(K, D, DIM, size, balance, fov_scale,
                                        new_camera)
            map_size = size
            print(f"[INFO] built remap table for {size[0]}x{size[1]}")

        out = cv2.remap(img, maps[0], maps[1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out_dir / path.name), out)
        written += 1

    print(f"[INFO] rectified {written} image(s) into {out_dir}")
    return written


def load_calibration(path):
    """Load K, D, DIM from an .npz written by 01_calibration/calibrate_fisheye.py."""
    data = np.load(path)
    return data["K"], data["D"], tuple(int(v) for v in data["DIM"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectify fisheye images to a pinhole projection (stage 02)."
    )
    parser.add_argument("--images", required=True,
                        help="Directory of fisheye images to rectify")
    parser.add_argument("--out", required=True,
                        help="Directory to write rectified images to")
    parser.add_argument("--calib", default=None,
                        help="calib.npz from stage 01. If omitted, the K/D/DIM "
                             "recorded in this file are used")
    parser.add_argument("--balance", type=float, default=0.0,
                        help="0 crops to all-valid pixels, 1 keeps the full FOV "
                             "(default: 0.0)")
    parser.add_argument("--fov-scale", type=float, default=0.8,
                        help="Divides the output focal length, widening the visible "
                             "field (default: 0.8)")
    parser.add_argument("--new-camera", choices=("estimated", "same-as-k"),
                        default="estimated",
                        help="How to pick the output camera matrix (default: "
                             "estimated)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calib:
        k, d, dim = load_calibration(args.calib)
    else:
        k, d, dim = K, D, DIM
        print("[INFO] using the K/D recorded in this script (no --calib given)")

    undistort_directory(args.images, args.out, k, d, dim,
                        balance=args.balance, fov_scale=args.fov_scale,
                        new_camera=args.new_camera)


if __name__ == "__main__":
    main()
