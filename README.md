# Structure from Motion

A cleaned and portfolio-ready version of a team project on **Structure from Motion (SfM)**, focused on **camera calibration**, **image undistortion**, **feature extraction and matching**, and **geometry estimation** for 3D scene reconstruction.

## Project Summary

This project explores a classical **multi-view computer vision** pipeline for recovering scene structure and camera motion from visual input.  
It covers several important stages of a practical Structure from Motion workflow, including calibration, distortion correction, local feature processing, and estimation-related steps for reconstruction.

I prepared this repository as a **cleaned GitHub version** of the original project for **portfolio presentation**, with unnecessary large files removed and folder names adjusted for cross-platform compatibility.

## Why This Project Matters

This project reflects my interest in:

- **Computer Vision**
- **3D Vision**
- **Multi-view Geometry**
- **Feature Matching**
- **Reconstruction Pipelines**
- **Vision-related Python Tooling**

It is especially relevant to roles related to:

- Computer Vision Intern / Working Student
- 3D Vision / Perception
- Robotics Vision
- Autonomous Systems Perception
- Visual Geometry / Reconstruction

## Skills / Keywords

Computer Vision, 3D Vision, Structure from Motion, Multi-view Geometry, OpenCV, SIFT, Feature Matching, Camera Calibration, Image Undistortion, Python, Visual Reconstruction, COLMAP-related reconstruction understanding

## Technical Scope

The project involves the following technical topics:

- camera calibration
- image undistortion
- feature extraction and matching
- SIFT-related processing
- estimation of scene / camera geometry
- visualization of intermediate and final outputs
- Python-based vision workflow organization

## Tools and Technologies

This project involved the use of several computer vision and 3D reconstruction related tools and concepts, including:

- **OpenCV** for camera calibration, image processing, and undistortion-related steps
- **SIFT-based feature processing** for local feature extraction and matching
- **Python** for scripting, workflow organization, and visualization utilities
- **Structure from Motion concepts** for multi-view geometry and reconstruction workflow understanding
- **COLMAP-related understanding / relevance** as an important reference tool in 3D reconstruction and SfM pipelines

## Pipeline Overview

The repository is organized around several key SfM stages:

1. **Calibration**  
   Estimation of camera parameters and calibration corner detection.

2. **Undistortion**  
   Correction of lens distortion using calibration results.

3. **Feature Processing**  
   Feature extraction and matching between views, including SIFT-related analysis.

4. **Estimation**  
   Geometry and motion estimation steps required for reconstruction.

5. **Visualization / Final Outputs**  
   Inspection of generated outputs and organization of final reconstruction-related artifacts.

## Repository Structure

```text
.
├── Literature/
├── Phase_2_Calibration/
├── Phase_3_Undistortion/
├── Phase_4_Apply_SIFT/
├── Phase_6_Estimate/
├── final/
├── image/
├── final_sfmcode_tutorial.txt
├── show_points.py
├── video_cut.py
├── Project_Report.pdf
├── sfm_presentation_final_version.pptx
└── README.md
```

## Folder Description

### `Phase_2_Calibration/`
Contains calibration-related outputs, corner detection results, and files used for camera parameter estimation.

### `Phase_3_Undistortion/`
Contains image undistortion related processing results after applying the estimated calibration parameters.

### `Phase_4_Apply_SIFT/`
Contains feature extraction / feature matching related material and visualizations.

### `Phase_6_Estimate/`
Contains estimation-related processing for scene structure and camera motion.

### `final/`
Contains selected final outputs and workspace material related to the reconstruction process.

### `Literature/`
Contains references and supporting literature used during the project.

## Key Files

### `show_points.py`
Python script for visualizing point data or reconstruction-related outputs.

### `video_cut.py`
Utility script for video preprocessing and sequence preparation.

### `Project_Report.pdf`
Final written report describing the project and results.

### `sfm_presentation_final_version.pptx`
Final presentation slides.

### `final_sfmcode_tutorial.txt`
Supplementary notes / instructions related to parts of the project workflow.

## Environment

This repository is mainly intended for portfolio and documentation purposes, but the workflow is related to a standard Python-based computer vision environment.

Typical dependencies include:

- Python 3
- OpenCV
- NumPy
- Matplotlib

Example installation:

```bash
pip install opencv-python numpy matplotlib
```

Depending on the exact script or reconstruction step, additional packages may be required.

## Usage

Clone the repository:

```bash
git clone git@github.com:JiahuiShen1998/Structure-from-Motion.git
cd Structure-from-Motion
```

Run individual scripts as needed, for example:

```bash
python show_points.py
```

Because this is a cleaned version of the original team project, some large raw inputs and intermediate artifacts were intentionally removed.

## My Contribution

For portfolio purposes, my contribution is best represented in the following aspects:

- understanding and organizing the overall Structure from Motion pipeline
- working with computer vision related processing steps such as calibration, undistortion, and feature-based matching
- using **OpenCV-related workflow concepts** in the project pipeline
- working with **SIFT-related feature extraction / matching**
- using **Python** for project scripts, visualization, and workflow support
- cleaning and restructuring the repository into a GitHub-friendly version
- improving project readability and presentation quality
- preparing the project for cross-platform usage by renaming incompatible folders
- documenting the project in a clearer and more professional way for academic and recruiting contexts

## What This Demonstrates

This repository demonstrates experience with:

- classical computer vision workflows
- multi-stage vision pipelines
- handling real project artifacts and intermediate outputs
- feature-based geometry processing
- cleaning and presenting technical projects in a professional format

## Recruiter-Friendly Summary

This project highlights my practical exposure to classical computer vision pipelines and 3D geometry-related workflows. It demonstrates my ability to understand, clean, organize, and present technical vision projects in a professional and GitHub-friendly format.

## Notes

- This repository is a **cleaned version** of the original team project.
- Large raw videos, compressed archives, and unnecessary intermediate files were removed to make the repository suitable for GitHub.
- Some folder names were renamed for compatibility with Windows and cross-platform development environments.
- The repository is intended mainly for **academic presentation** and **job portfolio use**.

## Future Improvements

Potential future improvements include:

- adding clearer setup instructions
- adding example inputs / outputs
- adding pipeline diagrams
- adding visualization screenshots directly in the README
- expanding documentation for each processing phase

## License

This repository is shared for academic and portfolio purposes.
