import sys
import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (9,7)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 300, 0.01)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
#+cv2.fisheye.CALIB_CHECK_COND
objp = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
#objp *= 100

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob("D:/AT_Master/Team Project/Calibrate-photo/calibrate*_RIGHT.png")
images = glob.glob('./images_calibration/calibrate*_RIGHT.png')

# Creat a directory for saving images with corner points
output_dir = './calibration_images_with_corners/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = 0
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("在图像索引中找到角点:", i)
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(13,13),(-1,-1),subpix_criteria)
        imgpoints.append(corners)

        # Plotting corner points on an image
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        # Displays the image after drawing the corners
        cv2.imshow('Corners', img)
        cv2.waitKey(500)  # 延迟 500 毫秒
        # Save the image to the specified directory
        output_filename = os.path.join(output_dir, f'corners_{i}.png')
        cv2.imwrite(output_filename, img)
        i += 1

print("3D points count:", len(objpoints))
print("2D points count:", len(imgpoints))

for i in range(len(imgpoints)):
    print(f"Coordinates of the corners of image {i}:")
    print(imgpoints[i].reshape(-1, 2))  # Flatten the array of corner points for viewing

N_OK = len(objpoints)
print("Number of valid images:", N_OK)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
if N_OK > 0:
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        None,#rvecs,
        None,#tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
    )
    print("RMS Error:", rms)
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
else:
    print("No valid images for calibration.")

'''
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    )


print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
'''



