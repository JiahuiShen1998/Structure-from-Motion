#sift 
import cv2 as cv
import numpy as np

# # define a function to detect the corner in image (SHI-TOMASI)
def detect_shi_corners(img, max_corners=2000, quality_level=0.01, min_distance=10):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    keypoints = [cv.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=50) for pt in corners]
    return keypoints


def sift_with_corners(img1_path, img2_path, num_matches=250, use_flann=True):
    # 1. imread the image
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    incresase_gray_1 = cv.equalizeHist(gray1)
    incresase_gray_2 = cv.equalizeHist(gray2)

    # 2. Shi-Tomasi(corner detection)
    corners1 =  detect_shi_corners(img1)
    corners2 =  detect_shi_corners(img2)

    # 3. SIFT (feature detection)
    sift = cv.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(gray1, None) # detect the key point
    # keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    keypoints1, descriptors1 = sift.compute(gray1, corners1) # detect the key point
    keypoints2, descriptors2 = sift.compute(gray2, corners2)

    print("the corner points number in image_1:", len(keypoints1))
    print("the corner points number in image_2:", len(keypoints2))

    green= (0,250,0)
    # img1_keypoints = cv.drawKeypoints(img1, keypoints1, None, color = green, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2_keypoints = cv.drawKeypoints(img2, keypoints2, None, color = green, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img1_keypoints = cv.drawKeypoints(img1, keypoints1, None, color = green, flags=2)
    img2_keypoints = cv.drawKeypoints(img2, keypoints2, None, color = green, flags=2)
    cv.imwrite("keypoints_img1.jpg", img1_keypoints)
    cv.imwrite("keypoints_img2.jpg", img2_keypoints)
    
    # 4. match the point(2 different ways)
    if use_flann:
        # FLANN 
        index_params = dict(algorithm=1, trees=4)  
        search_params = dict(checks=50)  # check number
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        #BFMatcher
        matcher = cv.BFMatcher(cv.NORM_L2)
    
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # 5. Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    good_matches = good_matches[:num_matches]

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # print("point in 1",pts1)
    # print("point in 2",pts2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, 5.0)  # use the fundamental matrix
    matches_mask = mask.ravel().tolist()
    
    ransac_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]] #(not sure for the ransac algorithm)
    ransac_matches = ransac_matches[:num_matches]
    
    # print("match matrix : ",H) # fundamental matrix

    # 6. draw the result
    matched_image = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,matchColor=green, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ransac_matches_image = cv.drawMatches(img1, keypoints1, img2, keypoints2, ransac_matches, None, matchColor=green,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # imshow and imwrite
    cv.imshow('SIFT with Corners(with ransac)', matched_image)
    cv.waitKey(0)
    cv.imwrite("match_with_corners_no_ransac.jpg", matched_image)
    cv.imwrite("with ransac.jpg",ransac_matches_image)
    cv.destroyAllWindows()

# Function
sift_with_corners('LMS02-right-undistorted/frame_95.jpg', 'LMS02-right-undistorted/frame_96.jpg', num_matches=250, use_flann=True)

