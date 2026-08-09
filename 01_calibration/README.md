# 01 — Fisheye calibration

Estimates the intrinsic matrix `K` and the four fisheye distortion coefficients `D`
from stills of a planar chessboard, using the Kannala–Brandt model via
`cv2.fisheye.calibrate`.

## Run

```bash
python calibrate_fisheye.py --images images_calibration/ --pattern 9x7 --out calib.npz
```

**Input** — stills of a 10×8-square chessboard (→ `9×7` *inner* corners) shot with the
target lens, all at the same resolution. The set used here was 29 images at 1920×1080.

**Output** — `calib.npz` holding `K`, `D`, `DIM` (the resolution `K` is valid for) and
the RMS reprojection error; one annotated image per successful detection under
`outputs/corners/`. `K` and `D` are also printed as pasteable `np.array(...)` literals.

Result for the lens in this project, at 1920×1080:

```
K = [[591.041,   0.000, 929.152],
     [  0.000, 594.439, 535.872],
     [  0.000,   0.000,   1.000]]
D = [-0.0052148, -0.0304902, 0.0098051, -0.00013177]
```

`DIM` matters as much as `K` — see stage 02, which scales `K` and leaves `D` alone.

## Three calibrations were run, and none of them converged

The team's working copy recorded three calibrations with the value they called `res`.
Reproducing the runs shows `res` is the RMS reprojection error returned by
`cv2.fisheye.calibrate`, in pixels:

| `res` (RMS px) | fx | cx | D |
|---|---|---|---|
| **204.5** (kept, and used for all published results) | 591.04 | 929.15 | −0.0052, −0.0305, +0.0098, −0.00013 |
| 227.8 | 824.98 | 953.34 | −0.1455, +0.1084, −0.1787, +0.0532 |
| 302.0 | 824.92 | 953.40 | −0.1461, +0.1108, −0.1823, +0.0550 |

A healthy chessboard calibration lands near **1 px**. These are two orders of magnitude
larger, and the focal length swings between ≈591 and ≈825 depending on which run you
look at — a 40% spread. This calibration did not converge; it terminated.

Re-running this script on the source image set reproduces that regime: 20 of 28 images
detect, and the fit lands at RMS 217.8 with fx = 772 — the same family as the recorded
227.8 / fx = 825 run. The script now warns when RMS exceeds 10 px.

## The set is ill-conditioned, and two views are why

`cv2.fisheye.calibrate` aborts outright on the full 20-image set:

```
(-215:Assertion failed) fabs(norm_u1) > 0 in function 'cv::internal::InitExtrinsics'
```

That assertion names neither the view nor the cause. `--check-cond` upgrades it to
`CALIB_CHECK_COND - Ill-conditioned matrix`, which is honest but still unnamed. A
leave-one-out sweep identifies the pair: **removing either `calibrate19_RIGHT.png` or
`calibrate23_RIGHT.png` lets the solver converge.** Which one you drop matters enormously:

| dropped | RMS | fx | fy |
|---|---|---|---|
| `calibrate19` | 217.8 | 772.5 | 773.3 |
| `calibrate23` | 387.9 | 191.5 | **31.5** |

An fy of 31 px is physically meaningless. Two nominally equivalent choices produce
intrinsics that differ by a factor of 25.

This is why the original code has `CALIB_CHECK_COND` commented out. It is not an
oversight — with the flag on, the calibration that produced every published result in
this project refuses to run at all.

Use `--exclude calibrate19_RIGHT.png,calibrate23_RIGHT.png` to drop them explicitly.

## Why the published intrinsics work anyway

Despite the above, the kept `K`/`D` rectify the 4K footage correctly: stage 02 uses
them to reproduce the team's archived undistorted frames to a mean absolute difference
of 1.00/255. The RMS is dominated by a few bad views while the parameters remain usable
for the bulk of the field. That is luck rather than method, and it is the honest reading
of this stage: the reconstruction downstream works, and the calibration feeding it is
not trustworthy enough to be used as a prior — which, as it happens, is exactly what
stage 05 does by registering a placeholder camera instead.

## Key parameters

- `--pattern 9x7` — **inner** corner count, one less than the square count in each
  direction. A 10×8-square board gives `9x7`. Wrong value → detection fails on every
  image with no other symptom.
- `SUBPIX_CRITERIA = (EPS + MAX_ITER, 300, 0.01)` with a `13×13` window — the window is
  the radius `cv2.cornerSubPix` searches for the true corner. On a fisheye frame the
  board edges curve, so a large window can lock onto a neighbouring edge; a small one
  gives up sub-pixel precision.
- `CALIB_RECOMPUTE_EXTRINSIC | CALIB_FIX_SKEW` — recompute each view's pose on every
  iteration (slower, much better conditioned), and force skew to zero, which is correct
  for any real sensor.
- `--check-cond` (off by default, matching the original) — on, OpenCV rejects an
  ill-conditioned set instead of absorbing the error into `K` and `D`. See above for why
  it cannot be left on for this dataset.
- `--exclude a.png,b.png` — drop specific views. The way out when the solver aborts.
- Object points are built with `np.mgrid` and **not** scaled by the physical square size.
  `K` and `D` are unaffected; recovered translation vectors are in units of squares.

## Failure modes

**Corner detection fails on a large fraction of the set, with no error.**
`findChessboardCorners` returns `False`, the image is skipped, and the only symptom is a
low final count, which the script now prints along with the rejected filenames. On the
source calibration set 20 of 28 images are accepted; on the stills committed here under
`../02_undistortion/data/chessboard_raw/`, only 11 of 29. On a fisheye lens this is
normal: the board bows near the frame border and the detector's quad-linking step cannot
chain the rows. Shoot 25–30 views expecting to lose a third, and keep the board away
from the extreme periphery.

**`fabs(norm_u1) > 0` assertion, or `Ill-conditioned matrix`.** One or more views are
degenerate. Confirm with `--check-cond`, find them by bisecting with `--exclude`, and
see the section above — on this dataset it is `calibrate19` and `calibrate23`.

**Every image fails.** Almost always the `--pattern` tuple: squares counted instead of
inner corners, or the two dimensions swapped. The detector gives no hint which it is.

**RMS is large (tens or hundreds of pixels).** The fit terminated without converging.
The script warns above 10 px. Usually a poorly conditioned view set; occasionally a
board whose physical geometry does not match `--pattern`.

**RMS is low but rectified images look wrong.** A low RMS only says the model fits the
corners that were detected, which may all sit near the frame centre. If no accepted view
puts the board near a corner of the frame, `D`'s higher-order terms are unconstrained
and extrapolate badly.

**"All calibration images must share one resolution".** One input is a different size.
Calibration is valid for a single `DIM` only; mixing resolutions produces a meaningless
`K`. The script names the offending file.

## Notes on the archived artifacts

The corner overlays under `outputs/corners/` came from the team's run. The chessboard
stills committed here under `../02_undistortion/data/chessboard_raw/` are **not** the set
the published `K`/`D` were fitted on — they are 1920×1080 and not circular fisheye, they
yield 11 detections rather than 20, and they calibrate to a completely different (and,
at RMS 1.74 px, far better-conditioned) solution with `fx ≈ 604`. The set that
reproduces the published regime is the team's `Calibratephoto/Calibrate-photo`, which is
not in this repository because of its size.
