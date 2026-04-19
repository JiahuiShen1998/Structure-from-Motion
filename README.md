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

## Technical Scope

The project involves the following technical topics:

- camera calibration
- image undistortion
- feature extraction and matching
- SIFT-related processing
- estimation of scene / camera geometry
- visualization of intermediate and final outputs
- Python-based vision workflow organization

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
