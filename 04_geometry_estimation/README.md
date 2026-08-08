# 04 — Geometry estimation

**There is no standalone script in this stage.** The directory exists to keep the
pipeline numbering continuous and to say where epipolar geometry and pose estimation
actually happen, because the original repository had an empty `Phase_6: Estimate/`
directory that was never populated and never explained.

## Where each piece lives

| Step | Implementation | Location |
|---|---|---|
| Fundamental matrix `F` per image pair | `cv2.findFundamentalMat`, RANSAC, 2.0 px, conf 0.99 | `05_reconstruction/sfm_pipeline.py` → `match_features_opencv()` |
| Inlier mask → verified match list | same call's `mask` | same function |
| Persisting verified geometry | `two_view_geometries` table, `config = 2` | same function |
| Initial pair selection | `mapper.find_initial_image_pair()` | `05_reconstruction/incremental_pipeline.py` → `init_reconstruction()` |
| Relative pose for the initial pair | `mapper.estimate_initial_two_view_geometry()` / `register_initial_image_pair()` | same |
| Absolute pose of each new view (PnP) | `mapper.register_next_image()` | `reconstruct_single_model()` |
| Triangulation (DLT) | `mapper.triangulate_image()` | same |

Everything below the OpenCV `F` estimate is `pycolmap` calling into COLMAP's C++
implementation. This repository drives it; it does not reimplement it.

## What `config = 2` means, and why `E`, `H`, `qvec`, `tvec` are placeholders

COLMAP's `two_view_geometries` table has columns for `F`, `E`, `H` and a relative pose
(`qvec`, `tvec`). The `config` integer says which of them are meaningful. Stage 05
writes `config = 2` — "calibrated/uncalibrated F verified, E and H not computed" — and
fills `E` and `H` with identity and the pose with `(1,0,0,0)`/`(0,0,0)`.

That is deliberate and safe: the mapper re-estimates two-view geometry itself during
initialisation and never reads those columns for a `config = 2` row. Writing a *wrong*
`config` — claiming `E` is valid when it holds an identity matrix — is what breaks
reconstruction, and does so with a confusing symptom (initialisation succeeds, then
every subsequent registration fails).

`E` is not computed here because it requires the intrinsics, and the intrinsics written
into the database at that point are a placeholder rather than the calibration from
stage 01. See the root README's limitations.

## Failure modes at this stage

**No good initial image pair found.** `pycolmap` reports
`IncrementalMapperStatus.NO_INITIAL_PAIR`. It needs a pair with enough verified inliers
*and* enough triangulation angle. Pure forward motion down a corridor gives plenty of
matches with almost no parallax, which fails the angle test. The pipeline already
handles this by relaxing constraints twice — halving `init_min_num_inliers`, then
halving `init_min_tri_angle` — in `incremental_mapper_pipeline()`. If it still fails,
the input sequence lacks sideways motion, not the parameters.

**Reconstruction splits into several disconnected models.** The mapper could not chain
registrations across a gap. Look at the per-pair inlier counts printed during matching:
a run of consecutive frames with single-digit inliers is the break. Usually motion blur
or a fast pan.

**`F` is estimated but degenerate.** If all correspondences lie on a plane, `F` is not
uniquely determined and RANSAC returns something that fits perfectly and means nothing.
`findFundamentalMat` can return a 9×3 matrix (three solutions) rather than 3×3; the
`F.shape != (3, 3)` check in `match_features_opencv()` drops the pair when that happens.
