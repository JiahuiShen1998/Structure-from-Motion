# References

Papers that informed the pipeline. PDFs are not tracked in this repository.

**Camera model and calibration**

- J. Kannala and S. Brandt, "A generic camera model and calibration method for
  conventional, wide-angle, and fish-eye lenses," *TPAMI* 28(8):1335–1340, 2006.
  The equidistant fisheye model implemented by `cv2.fisheye`.
- Z. Zhang, "A flexible new technique for camera calibration," *TPAMI*
  22(11):1330–1334, 2000. The planar-target calibration procedure behind
  `findChessboardCorners` + `calibrate`.
- M. Kedzierski and A. Fryskowska, "Precise method of fisheye lens calibration,"
  *ISPRS Congress*, Beijing, 2008.
- E. Schwalbe, "Geometric modelling and calibration of fisheye lens camera systems,"
  *2nd Panoramic Photogrammetry Workshop*, 2005.

**Structure-from-Motion**

- J. L. Schönberger and J.-M. Frahm, "Structure-from-Motion Revisited," *CVPR* 2016.
  COLMAP's incremental mapper — the algorithm `05_reconstruction/` drives.
- P. Moulon, P. Monasse and R. Marlet, "Global fusion of relative motions for robust,
  accurate and scalable structure from motion," *ICCV* 2013. The global alternative,
  not used here.
- S. Bianco, G. Ciocca and D. Marelli, "Evaluating the performance of structure from
  motion pipelines," *Journal of Imaging* 4(8):98, 2018.

**Features and robust estimation**

- T. Tuytelaars and K. Mikolajczyk, "Local invariant feature detectors: a survey,"
  *FnT in Computer Graphics and Vision* 3(3):177–280, 2008.
- P. Torr and A. Zisserman, "MLESAC: A new robust estimator with application to
  estimating image geometry," *CVIU* 78(1):138–156, 2000.

**Fisheye 3D reconstruction**

- C. Ma, L. Shi, H. Huang and M. Yan, "3D reconstruction from full-view fisheye
  camera," arXiv:1506.06273, 2015.

**Consulted implementations**

- H. Venkataraman, *3D Reconstruction using Structure from Motion*, MIT licence,
  https://github.com/harish-vnkt/structure-from-motion. A pure OpenCV/NumPy incremental
  SfM implementation, read as a reference for the registration and triangulation loop.
  No code from it is used here — this pipeline delegates that loop to `pycolmap`.
- COLMAP's `scripts/python/database.py` and `read_write_model.py`, BSD-3-Clause, vendored
  unmodified under `05_reconstruction/third_party/`.
- The `pycolmap` `custom_incremental_pipeline.py` / `custom_bundle_adjustment.py`
  examples, whose structure `05_reconstruction/incremental_pipeline.py` and
  `bundle_adjustment.py` follow.
