# Defects found, and what was done about them

Found while documenting the pipeline against its archived outputs. Fixes are limited to
defects — no algorithm or numeric parameter was changed, so the documented settings and
the archived results still describe the same computation.

## Fixed

| File | Defect | Fix |
|---|---|---|
| `05_reconstruction/sfm_pipeline.py` | A stray `")` at the end of a `print` made the file **unparseable** — the pipeline entry point could not be imported or run at all. Introduced when the file's Chinese comments and messages were translated to English; the pre-translation revision in the repository history is clean. | Removed the stray quote. |
| `05_reconstruction/incremental_pipeline.py` | `f"{timestamp:010d}"` applied to `time.time() * 1000`, a float. The `d` format spec rejects floats with a `ValueError`. Only reachable when `snapshot_images_freq > 0`, which is not the default, so it had never fired. | Cast to `int` before formatting. |
| `03_feature_matching/match_shi_sift.py` | `good_matches[:250]` truncated **before** RANSAC and **without** sorting, so the geometric check saw an arbitrary subset in keypoint order rather than the best matches. | Sort by descriptor distance; drop the pre-RANSAC truncation entirely. RANSAC now sees every match that passed the ratio test, and `--num_matches` caps the drawing only. On the archived frame pair this takes RANSAC's input from an arbitrary 250 to all 559 candidates, of which 350 are inliers. |
| `01_calibration/calibrate_fisheye.py` | `if _img_shape == None` instead of `is None`. | Replaced by an explicit resolution check that names the offending file. |
| `01_calibration/calibrate_fisheye.py` | A degenerate calibration view made `cv2.fisheye.calibrate` abort with `(-215:Assertion failed) fabs(norm_u1) > 0 in InitExtrinsics`, which says nothing about which view or what to do. | Caught and re-raised with the diagnosis procedure, plus an `--exclude` flag to drop the offending views and `--check-cond` to confirm. |
| `01_calibration/calibrate_fisheye.py`, `tools/extract_frames.py`, `tools/view_pointcloud.py`, `05_reconstruction/sfm_pipeline.py` | Input and output paths were hardcoded to one developer's machine (`D:/AT_Master/...`, `F:/team project/...`). | Replaced with argparse CLIs. `sfm_pipeline.py` resolves COLMAP from `--colmap`, then `$COLMAP_EXE`, then `PATH`. |
| `tools/extract_frames.py` | `f"frame_{n:02d}"` collides past 100 frames; the recorded runs saved up to `frame_99`, one short of the bug. | Padding width derived from the frame count. |
| `.gitignore` | The file had been written twice and the seam merged two patterns into `*.pyc*.zip`, matching neither. | Rewritten. |

## Not fixed

| File | Issue | Why it stands |
|---|---|---|
| `05_reconstruction/sfm_pipeline.py:251-254` | The stage-01 calibration never reaches the reconstruction. A placeholder `SIMPLE_RADIAL` camera is registered instead — 3840×2160, f = 1500, principal point at the exact image centre, `prior_focal_length=True`. The scaled calibrated focal would be ≈1182 px and the calibrated principal point is not centred. The shipped `database.db` confirms the placeholder is what was used. | Changing it changes the reconstruction, which would invalidate every archived result the documentation cites. It is the repository's most substantive limitation and is described as such in the root README. |
| `05_reconstruction/sfm_pipeline.py:291-296, 73` | Lossy descriptor round-trip: L2-normalise → ×512 → clip to 255 → `uint8`, decoded as `/512`. Any component above ≈0.498·‖d‖ is clipped flat, and COLMAP's own convention does not normalise this way, so these descriptors are not comparable to COLMAP-extracted ones. | Same reason: it would change the matches and therefore the model. |
| `05_reconstruction/sfm_pipeline.py:20, 93` | `match_features_opencv(..., vis_path)` creates the directory but never writes match visualisations. Dead parameter. | Harmless; removing it would change the call signature for no gain. |
| `01_calibration/calibrate_fisheye.py` | `objp *= square_size` is commented out, so object points are in units of chessboard squares. `K` and `D` are unaffected; recovered translation vectors are on an arbitrary scale. | Matches how the calibration was actually run, and nothing downstream consumes those translations. |
| `01_calibration/calibrate_fisheye.py` | `cv2.fisheye.CALIB_CHECK_COND` is off by default. | This is not an oversight. With it on, the calibration set used for the published intrinsics is rejected outright as ill-conditioned — see the stage README. It is exposed as `--check-cond`. |

## Fixed before this repository existed

Kept because the symptom is worth recognising. An early revision of the matching
experiment built the second point set with the wrong index:

```python
pts2 = np.float32([keypoints2[m.queryIdx].pt for m in good_matches])   # wrong
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])   # correct
```

`DMatch.queryIdx` indexes the first keypoint list, `trainIdx` the second. Using
`queryIdx` against `keypoints2` pairs each point in image 1 with an unrelated point in
image 2. It does not crash — the indices are usually in range — RANSAC simply keeps
almost nothing regardless of threshold, which reads exactly like a parallax problem and
sends you looking in the wrong place. The result was also stored in a variable named
`H`, as though it were a homography, which made the mistake harder to see.
