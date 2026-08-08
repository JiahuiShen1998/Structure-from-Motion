# 03 — Feature matching (standalone experiment)

A two-frame matching experiment used to settle the detector/descriptor combination and
to see what RANSAC actually removes, before matching was wired into the full pipeline.
It is not on the production path: stage 05 re-extracts and re-matches with its own
settings.

## Run

```bash
python match_shi_sift.py \
  --img1 ../05_reconstruction/workspace/images/frame_95.jpg \
  --img2 ../05_reconstruction/workspace/images/frame_96.jpg \
  --output_dir outputs --use_flann
```

**Input** — two rectified frames from the same sequence.
**Output** — into `--output_dir`: `keypoints_img1.jpg`, `keypoints_img2.jpg`,
`matches_before_ransac.jpg`, `matches_after_ransac.jpg`. For 4K inputs the match
drawings are 7680×2160 side-by-side composites.

## What it actually does

The detector and the descriptor are deliberately not the same algorithm:

1. `cv.goodFeaturesToTrack` — Shi-Tomasi corners, `maxCorners=2000`,
   `qualityLevel=0.01`, `minDistance=10`.
2. Those corners are wrapped as `cv.KeyPoint`s with a fixed `size=50`.
3. `sift.compute()` — SIFT **descriptors only**, at those locations and that fixed
   scale. `detectAndCompute` is not used.

The point is spatial spread: Shi-Tomasi's `minDistance` enforces it, SIFT's DoG
detector does not. The cost is that scale is hardcoded for every keypoint, so scale
invariance is gone. Between adjacent video frames the scale change is small enough that
this holds; across a wider baseline it would not. Stage 05 therefore uses plain
`detectAndCompute`.

## How the parameters were arrived at

An earlier revision of this experiment survives in the team's working copy. The
difference between the two is the actual tuning record:

| Parameter | first pass | kept |
|---|---|---|
| `max_corners` | 180 | **2000** |
| `min_distance` | 15 | **10** |
| KeyPoint `size` | 10 | **50** |
| matcher default | BFMatcher | **FLANN** |
| FLANN `trees` | 8 | **4** |
| Lowe ratio | 0.5 | **0.7** |
| RANSAC threshold | 2.0 px | **5.0 px** |

Reading it as a story: 180 corners at `size=10` is far too few keypoints described over
far too small a patch, so descriptors were not distinctive and the ratio test at 0.5 —
already aggressive — threw away most of what was left. The fix went in both directions
at once: an order of magnitude more corners, a 5× larger descriptor patch, and a
looser ratio. The RANSAC threshold was then relaxed from 2.0 to 5.0 px because the
tighter value was rejecting inliers that were genuine but imprecisely localised, which
is expected once keypoints sit at a fixed scale rather than at their true DoG scale.

Note that stage 05 later goes the other way on both knobs — ratio 0.6, RANSAC 2.0 px —
because it uses real `detectAndCompute` keypoints, which are better localised and
support a stricter geometric test.

## The bug that was fixed between the two revisions

The first pass built the second point set with the wrong index:

```python
pts2 = np.float32([keypoints2[m.queryIdx].pt for m in good_matches])   # wrong
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])   # correct
```

`DMatch.queryIdx` indexes the *first* keypoint list, `trainIdx` the second. Using
`queryIdx` against `keypoints2` pairs each point in image 1 with an unrelated point in
image 2, so the correspondences handed to `findFundamentalMat` are noise. It does not
crash — the indices are usually in range — it just silently produces a meaningless `F`
and a near-empty inlier set. The symptom is that RANSAC keeps almost nothing no matter
how the threshold is set, which looks exactly like a parallax problem and sends you
looking in the wrong place. The variable holding the result was also named `H`, as
though it were a homography, which made the mistake harder to see; it is `F` now.

## Key parameters

- **`--quality_level` (0.01)** — a corner is kept if its Shi-Tomasi score is ≥ 1% of the
  best in the image. Raising it thins low-contrast regions first.
- **`--min_distance` (10)** — minimum spacing between corners. This is what buys spatial
  spread; without it corners pile onto the highest-contrast object.
- **KeyPoint `size` (50)** — descriptor patch scale, fixed. Too small and the descriptor
  is not distinctive; too large and the patch spans a depth discontinuity, so it changes
  with viewpoint.
- **`--ratio_threshold` (0.7)** — Lowe's ratio. Rejects *ambiguous* matches, not wrong
  ones.
- **FLANN** `algorithm=1` (KD-tree), `trees=4`, `checks=50` — `checks` is the
  accuracy/speed dial. At 128-D, KD-trees are past the dimensionality where they beat
  brute force by much; omit `--use_flann` to get exact `NORM_L2` matching as a
  cross-check when the match count looks wrong.
- **`--ransac_thresh` (5.0 px)** — maximum distance from the epipolar line for an inlier.

## Failure modes

**Matches look dense and plausible but RANSAC keeps almost none.** Either too little
parallax (near-pure rotation makes the epipolar geometry degenerate and `F` unstable —
the tell is that the inlier set changes between runs on identical input), or the
index bug above. Check the index usage first; it is cheaper to rule out.

**`cv.error` inside `knnMatch`.** FLANN needs `float32` descriptors. If one image
yielded no corners, `descriptors` is `None`; the script raises a named `RuntimeError`
before this point rather than letting OpenCV fail obscurely.

**Matches concentrate in one corner of the frame.** `min_distance` too small, or one
dominant high-contrast object. Poorly spread correspondences give a badly conditioned
`F` even when the inlier count looks healthy.

## Remaining defect

`good_matches[:num_matches]` truncates to 250 **before** RANSAC and **without** sorting
by descriptor distance, so the 250 retained are an arbitrary subset in matcher order
rather than the 250 best. Carried over from the original experiment and not fixed. See
[`../docs/known_bugs.md`](../docs/known_bugs.md).
