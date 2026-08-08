# Known defects

Found while documenting the code. **None of these have been fixed** — the code is kept
as it was run so that the documented parameters and the archived outputs still describe
the same program. Ordered by severity.

| # | File | Line | Defect |
|---|---|---|---|
| 1 | `05_reconstruction/sfm_pipeline.py` | 429 | Stray `")` at the end of a `print`. **The file does not parse.** Introduced when the file's Chinese comments and messages were translated to English; the earlier revision in the repository history is syntactically clean. One-character fix |
| 2 | `05_reconstruction/sfm_pipeline.py` | 251–254 | The stage-01 calibration never reaches the reconstruction. A hardcoded `SIMPLE_RADIAL` camera (3840×2160, f = 1500, principal point at the exact image centre, `prior_focal_length=True`) is registered instead. The scaled calibrated focal would be ≈1182 px, and the calibrated principal point is not centred. The shipped `database.db` confirms the placeholder is what was used |
| 3 | `05_reconstruction/incremental_pipeline.py` | 41 | `f"{timestamp:010d}"` applied to a `float` raises `ValueError`. Only reachable when `snapshot_images_freq > 0`, which is not the default |
| 4 | `03_feature_matching/match_shi_sift.py` | — | ~~unused `equalizeHist`~~ **fixed** in the refactor; the dead path was removed |
| 5 | `03_feature_matching/match_shi_sift.py` | 162 | `good_matches[:num_matches]` truncates to 250 **before** RANSAC and **without** sorting by descriptor distance. The retained 250 are an arbitrary subset in matcher order, not the best 250 |
| 6 | `05_reconstruction/sfm_pipeline.py` | 291–296, 73 | Lossy descriptor round-trip: L2-normalise → ×512 → clip to 255 → `uint8`, decoded as `/512`. Any component above ≈0.498·‖d‖ is clipped flat. COLMAP's own convention does not normalise this way, so these descriptors are not comparable to COLMAP-extracted ones |
| 7 | `05_reconstruction/sfm_pipeline.py` | 20, 93 | `match_features_opencv(..., vis_path)` creates the directory but never writes match visualisations. Dead parameter |
| 8 | `01_calibration/calibrate_fisheye.py` | 31 | `if _img_shape == None` should be `is None`. Works for a tuple, but fails for any type with a custom `__eq__` |
| 9 | `01_calibration/calibrate_fisheye.py` | 15 | `objp *= 100` is commented out, so object points are in units of chessboard squares. `K` and `D` are unaffected; the returned translation vectors are on an arbitrary scale |

## Also worth knowing (not bugs)

- `cv2.fisheye.CALIB_CHECK_COND` is commented out in the calibration flags. With it
  enabled, OpenCV aborts and names the view that makes the system ill-conditioned.
  Disabled, the fit absorbs that view's error silently.
- `05_reconstruction/sfm_pipeline.py` carries a UTF-8 BOM. Harmless in Python 3, but it
  shows up as `﻿` in front of the first `import` in diffs.
- `set_colmap_path()` returns `None` rather than raising when `colmap.exe` is missing,
  and `run_pipeline` then returns early. The sparse reconstruction is already on disk at
  that point, so a failed dense stage looks like a successful run.
- Line numbers refer to the files at their current paths. `match_shi_sift.py` is the
  refactored version of the original `_sift_12.11.2024_.py`; two of that script's
  defects (the unused `equalizeHist` and an output filename containing a space) are
  already fixed there.
- Fixed earlier, kept here because the symptom is worth recognising: an initial
  revision of the matching experiment built `pts2` with `m.queryIdx` instead of
  `m.trainIdx`, pairing each point in image 1 with an unrelated point in image 2. It
  does not crash; RANSAC just keeps almost nothing, which is easy to misread as a
  parallax problem. See `03_feature_matching/README.md`.
