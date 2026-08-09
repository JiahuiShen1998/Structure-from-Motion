# 05 — Reconstruction

The production pipeline. Extracts SIFT into a COLMAP-format SQLite database, matches
every image pair and verifies each with a fundamental matrix, runs incremental
registration + triangulation + bundle adjustment through `pycolmap`, then shells out to
the COLMAP binary for dense stereo.

This stage does **not** consume the output of stage 03 — it re-extracts and re-matches
with its own settings (ratio 0.6 instead of 0.7, RANSAC 2.0 px instead of 5.0, plain
`detectAndCompute` instead of Shi-Tomasi seeding).

## Run

```bash
python sfm_pipeline.py --workspace workspace/ --colmap /path/to/colmap.exe

# sparse only, no windows -- works over SSH
python sfm_pipeline.py --workspace workspace/ --skip-dense --no-show
```

Put undistorted images in `<workspace>/images/` first. Dense reconstruction needs the
CUDA build of COLMAP 3.11.0; the executable is resolved from `--colmap`, then
`$COLMAP_EXE`, then `PATH`, and if none of them find it the pipeline stops after sparse
reconstruction rather than failing.

**Outputs**, all under `<workspace>/`:

```
database.db              keypoints, descriptors, matches, two_view_geometries
visualizations/          one keypoint overlay per input image
sfm/0/                   sparse model — cameras.bin, images.bin, points3D.bin
sfm/dense/fused.ply      dense point cloud
INFO.log                 pycolmap log
```

Two Open3D windows open in sequence. **The dense stage does not start until you close
the sparse window** — the visualiser call blocks.

## Stages inside `sfm_pipeline.py`

| # | Function | What it does |
|---|---|---|
| 1 | `extract_features_opencv_to_db` | `SIFT_create(nfeatures=8192)`, `detectAndCompute` per image, writes keypoints + descriptors |
| 2 | `match_features_opencv` | Exhaustive FLANN matching, ratio test, `findFundamentalMat` verification, writes `matches` and `two_view_geometries` |
| 3 | `incremental_pipeline.run_incremental_sfm` | Initial pair, PnP registration, triangulation, local/global BA |
| 4 | `show_sparse_pointcloud` | Reads `points3D.bin`, renders in Open3D |
| 5 | `dense_reconstruction` | COLMAP `image_undistorter` → `patch_match_stereo` → `stereo_fusion` |
| 6 | `show_stereo_result` | Renders `fused.ply` |

`incremental_pipeline.py` and `bundle_adjustment.py` are Python ports of COLMAP's C++
incremental mapper, following the structure of the upstream `pycolmap` example scripts.
They exist so the registration and BA loop is inspectable and instrumentable rather
than a single opaque `pycolmap.incremental_mapping()` call.

`third_party/database.py` and `third_party/read_write_model.py` are unmodified upstream
COLMAP scripts (BSD-3-Clause, ETH Zürich / UNC Chapel Hill).

## Parameters

| Parameter | Value | Effect |
|---|---|---|
| `nfeatures` | 8192 | Upper bound on SIFT keypoints per image, not a target. Actual output here averaged 1527 (510–2491), so the cap never binds — raising it changes nothing |
| `ratio_test` | 0.6 | Lowe's ratio. Tightened from 0.55 between pipeline revisions. Lower → fewer, cleaner matches; the risk is starving weakly-textured pairs below `min_inliers` |
| `ransac_thresh` | 2.0 px | Max distance from the epipolar line for an inlier. At 4K this is tight; loosening it inflates inlier counts with matches that later fail triangulation |
| `min_inliers` | 5 | Minimum verified matches to keep a pair. Very low — see failure modes |
| FLANN | `trees=5, checks=50` | Approximate NN accuracy/speed. `checks` is the dial |
| camera model | `SIMPLE_RADIAL`, 3840×2160, f=1500, pp=(1920,1080) | **Placeholder, not the stage-01 calibration.** See limitations |
| `max_image_size` | 5000 | `image_undistorter` cap; above the 3840 input, so no downscale |
| `geom_consistency` | true | Cross-checks depth maps between views. Roughly doubles stereo time, removes most fliers |
| `num_threads` | 16 | `stereo_fusion` CPU threads |

## Recorded run

48 frames (`frame_52.jpg` … `frame_99.jpg`) at 3840×2160, indoor sequence:

- 1527 keypoints per image on average (510–2491)
- 1003 verified pairs out of 1128 exhaustive pairs (88.9%)
- 67.3 inliers per verified pair on average (7–437)

Reported timings for this input: matching ~120 s, sparse ~180 s, dense 2–3 h on a
current GPU and roughly 17 h on an older one.

## Failure modes

**`ModuleNotFoundError: pycolmap`.** The dense and sparse stages have different
dependency footprints; `pip install -r requirements.txt` covers the Python side, but
COLMAP itself is a separate binary. See the repository root README.

**`No good initial image pair found`.** Covered in
[`../04_geometry_estimation/README.md`](../04_geometry_estimation/README.md).

**Dense reconstruction silently does nothing.** `set_colmap_path()` returns `None` when
COLMAP cannot be resolved, and `run_pipeline` returns early with a message naming the
path it tried. The sparse result is already on disk at that point, so a skipped dense
stage can read as success if you are not watching the log.

**`patch_match_stereo` fails or produces empty depth maps.** Needs the **CUDA** build of
COLMAP and a visible GPU at `gpu_index 0`. The CPU build has no PatchMatch stereo at
all. Check the subprocess stderr, which is captured and printed by `run_colmap_command`.

**Pairs kept with 7 inliers.** `min_inliers = 5` admits pairs whose frames barely
overlap. They do not corrupt the model — the mapper filters them — but they add
matching time and noise to the pair statistics. A healthy adjacent-frame pair here is
in the hundreds.

**Reprojection errors climb during global BA.** `run_global_ba` anchors gauge freedom
by fixing the first registered image's pose entirely and one coordinate of the second's
position — 7 DOF in total. If the first two registered images are nearly coincident,
that anchor is weak and the whole model can drift.

**Keypoint count far below `nfeatures`.** Expected. SIFT finds what the image contains;
`nfeatures` only truncates. A count near 500 means a low-texture frame, which is
worth knowing before blaming the matcher.

## Known limitation carried by this stage

The stage-01 calibration never reaches the reconstruction. A placeholder
`SIMPLE_RADIAL` camera with f = 1500 is registered instead; the calibrated fisheye
focal scaled to 4K would be ≈1182 px. The shipped `database.db` confirms the
placeholder is what was used. COLMAP self-calibrates away from it during BA, so the
result is not wrong so much as unnecessarily unconstrained.
