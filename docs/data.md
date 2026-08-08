# Data

The original repository tracked 7.06 GB of raw media, including eight MP4 files between
730 MB and 803 MB. That exceeds GitHub's 100 MB per-file hard limit and made the
repository effectively unclonable, so it was removed from git history. This file records
what was there and how to regenerate the derived data.

## Removed from history

| Path | Size | Contents |
|---|---|---|
| `Recordings/*.mp4` | 6.5 GB | 12 fisheye clips: 4 views (front/back/left/right) × 3 scenes (`114_ParkingDeckStraight`, `121_LMS01`, `122_LMS02`) |
| `image/LMS01-front.zip`, `LMS02-front.zip` | 217 MB | Extracted front-camera frames |
| `Phase_2: Calibration/LMS01-right-undistorted.zip` | 79 MB | 40 undistorted frames, 3840×2160 |
| `Phase_2: Calibration/rectified_right.zip` | 14 MB | 40 rectified frames, 1920×1080 |
| `Team_project_report.zip` | 38 MB | LaTeX source of the project report — the built PDF is in this directory |
| `sfm_presentation_final_version.pptx` | 33 MB | Final presentation |
| `reference.zip` | 2.4 MB | Duplicate of four papers listed in `references.md` |
| `Literature/*.pdf` | 32 MB | 11 papers, indexed in `references.md` |

The scene names are lab identifiers: `LMS01` / `LMS02` are indoor rooms at the LMS
chair, `ParkingDeckStraight` is the outdoor parking deck sequence.

## What is still in the repository

- `05_reconstruction/workspace/images/` — 48 frames, `frame_52.jpg` … `frame_99.jpg`,
  3840×2160, the exact input to the recorded reconstruction run
- `05_reconstruction/workspace/database.db` — the COLMAP database produced from them
- `01_calibration/outputs/corners/` — 20 annotated corner detections, 1920×1080
- `02_undistortion/data/chessboard_raw/` — 29 raw chessboard stills, 1920×1080
- `03_feature_matching/outputs/` — two match visualisations, 7680×2160

## Regenerating

```bash
# 1. frames from video, every 10th
python tools/extract_frames.py --video 121_LMS01_right.mp4 --out frames/ --interval 10

# 2. calibrate on the chessboard stills that are still in the repo
python 01_calibration/calibrate_fisheye.py \
    --images 02_undistortion/data/chessboard_raw/ --out calib.npz

# 3. rectify
python 02_undistortion/undistort_fisheye.py \
    --calib calib.npz --images frames/ --out undistorted/

# 4. reconstruct
cp undistorted/* 05_reconstruction/workspace/images/
python 05_reconstruction/sfm_pipeline.py \
    --workspace 05_reconstruction/workspace --colmap /path/to/colmap.exe
```

Step 3 needs the distortion coefficients `D`. Only `K` was recorded in the project
report; `D` has to be re-derived by running step 2.
