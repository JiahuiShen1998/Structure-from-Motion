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

## Three calibrations were run; the one above is the one that was kept

The team's working copy recorded three converged calibrations with their residuals.
They are kept here because the spread is instructive:

| residual | fx | cx | D |
|---|---|---|---|
| **204.5** (kept) | 591.04 | 929.15 | −0.0052, −0.0305, +0.0098, −0.00013 |
| 227.8 | 824.98 | 953.34 | −0.1455, +0.1084, −0.1787, +0.0532 |
| 302.0 | 824.92 | 953.40 | −0.1461, +0.1108, −0.1823, +0.0550 |

Two things are worth reading off this table. First, the focal length is not weakly
determined — it lands at either ≈591 or ≈825 depending on which corner set the fit
converged on, a 40% spread, so residual alone is a thin basis for choosing. Second,
the two higher-residual solutions have distortion coefficients that alternate sign and
are 20–30× larger; that is the signature of an over-fitted, non-monotonic θ_d(θ). Those
two are not merely worse, they are unusable downstream:
`estimateNewCameraMatrixForUndistortRectify` collapses on them and returns a degenerate
camera, producing an almost entirely black rectified image. The kept solution is the
only one of the three that survives stage 02.

The `0.01` / `0.001` annotations beside the two rejected runs correspond to the
`subpix_criteria` epsilon. Tightening sub-pixel refinement made the fit *worse* here,
which is the opposite of the usual expectation and is worth knowing before spending
time on that knob.

## Key parameters

- `CHECKERBOARD = (9, 7)` — **inner** corner count, i.e. one less than the square count
  in each direction. A 10×8-square board gives `(9, 7)`.
- `subpix_criteria = (EPS + MAX_ITER, 300, 0.01)` with a `13×13` refinement window —
  the window is the radius `cv2.cornerSubPix` searches for the true corner. On a
  fisheye frame the board edges curve, so a large window can lock onto a neighbouring
  edge; a small one gives up sub-pixel precision.
- `calibration_flags = CALIB_RECOMPUTE_EXTRINSIC | CALIB_FIX_SKEW` — recompute each
  view's pose on every iteration (slower, much better conditioned), and force the skew
  term to zero, which is correct for any real sensor.
- `CALIB_CHECK_COND` is **commented out**. With it on, OpenCV raises as soon as any
  single view makes the system ill-conditioned, and names the offending index. With it
  off — as here — the fit proceeds and quietly absorbs that view's error into `K` and
  `D`. Turn it back on if the RMS looks fine but rectification does not.
- Object points are built with `np.mgrid` and **not** scaled by the physical square
  size (`objp *= 100` is commented out). `K` and `D` are unaffected, but the
  translation vectors come out in units of chessboard squares rather than millimetres.

## Failure modes

**Corner detection fails on a large fraction of the set, with no error.**
`findChessboardCorners` returns `False`, the image is skipped, and the only symptom is
a low final count. Here 20 of 29 images were accepted; indices 01, 02, 05, 10, 12, 13,
15, 18 and 28 were rejected. On a fisheye lens this is normal: the board bows near the
frame border and the detector's quad-linking step cannot chain the rows. Shoot 25–30
views expecting to lose a third, and keep the board away from the extreme periphery.

**Every image fails.** Almost always the `CHECKERBOARD` tuple — squares counted instead
of inner corners, or the two dimensions swapped. Check by drawing the raw image first;
the detector gives no hint which it is.

**RMS is low but rectified images look wrong.** A low RMS only says the model fits the
detected corners, which may all be clustered in the frame centre. Look at the spatial
coverage of the accepted views: if no accepted view puts the board near a corner of the
frame, `D`'s higher-order terms are unconstrained and extrapolate badly. Re-enable
`CALIB_CHECK_COND` to find the views that are dragging the fit.

**`assert _img_shape == img.shape[:2]` trips.** One input is a different resolution.
Calibration is only valid for a single `DIM`; mixing resolutions silently produces a
meaningless `K` if the assert is removed.

## Notes

The corner images under `outputs/corners/` are named after their source frame
(`corners_calibrate07_RIGHT.png`), while the script writes a running counter
(`corners_0.png`, `corners_1.png`, …). Both naming schemes exist in the team's working
copy, so the committed set was renamed by hand after the fact.

**Open question.** The chessboard stills in `../02_undistortion/data/chessboard_raw/`
are 1920×1080 and are *not* circular fisheye — the image fills the frame and the
distortion is moderate. The 4K video frames that stage 02 rectifies *are* circular
fisheye, with a black surround. Those are different fields of view, so this set of
stills is probably not the one the recorded `K`/`D` were fitted on, even though the
recorded values demonstrably rectify the 4K frames correctly. Re-running this script on
the committed stills will not reproduce the `K` above.
