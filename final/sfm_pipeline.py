import os
import cv2
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import subprocess

from pathlib import Path
from pycolmap import logging

# Custom modules
import read_write_model as rw
import incremental_pipeline
from database import COLMAPDatabase, array_to_blob, image_ids_to_pair_id



def match_features_opencv(database_path, vis_path, ratio_test=0.6, ransac_thresh=2.0, min_inliers=5):
    """
    Match features via OpenCV & save to COLMAP Database.

    Args:
        database_path (str): Path to the COLMAP database.
        vis_path (str): Path to save match visualization.
        ratio_test (float): Ratio test threshold for KNN matching.
        ransac_thresh (float): RANSAC inlier threshold for Fundamental matrix.
        min_inliers (int): Minimum inliers for a valid image pair.

    Steps:
        1. Read images & features from DB.
        2. Match features pairwise (KNN + ratio test).
        3. Geometric verification with RANSAC.
        4. two_view_geometries & matches / Save verified matches to DB.
    """
    print("# 2. Feature Matching (OpenCV)")

    # 1) Open DB
    db = COLMAPDatabase.connect(database_path)

    # 2) Read images table
    rows = db.execute("SELECT image_id, name FROM images ORDER BY image_id").fetchall()
    image_list = [(r[0], r[1]) for r in rows]
    print(f"Total {len(image_list)} images in DB.")

    # 3) Load keypoints & descriptors
    keypoints_table = {}
    descriptors_table = {}

    #   - keypoints
    rows_keypoints = db.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall()
    for (img_id, r, c, blob) in rows_keypoints:
        try:
            kpts = np.frombuffer(blob, dtype=np.float32).reshape(r, c)
            keypoints_table[img_id] = kpts
        except Exception as e:
            print(f"图像ID {img_id} 关键点加载失败 / Failed to load keypoints: {e}")
            continue

    #   - descriptors
    rows_desc = db.execute("SELECT image_id, rows, cols, data FROM descriptors").fetchall()
    for (img_id, r, c, blob) in rows_desc:
        expected_cols = 128
        actual_size = len(blob)
        # each descriptor is 128 bytes
        if actual_size % expected_cols != 0:
            print(f"警告: 图像ID {img_id} 描述符数据有误 / Descriptor size mismatch: {actual_size}")
            continue
        r = actual_size // expected_cols
        try:
            desc_uint8 = np.frombuffer(blob, dtype=np.uint8).reshape(r, expected_cols)
            desc_float = desc_uint8.astype(np.float32) / 512.0  # 逆向缩放 / reverse scale
            descriptors_table[img_id] = desc_float
        except ValueError as e:
            print(f"image ID {img_id} Descriptor reshape failed: {e}")
            continue

    # 4) Clear old records
    db.execute("DELETE FROM two_view_geometries")
    db.execute("DELETE FROM matches")

    # 5) Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 6) Pairwise matching
    total_pairs = len(image_list) * (len(image_list) - 1) // 2
    print(f"Matching {total_pairs} image pairs...")

    os.makedirs(vis_path, exist_ok=True)
    successful_matches = 0

    for idx, ((id1, name1), (id2, name2)) in enumerate(itertools.combinations(image_list, 2), 1):
        desc1 = descriptors_table.get(id1, None)
        desc2 = descriptors_table.get(id2, None)
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            continue

        # 6.2 KNN match
        try:
            knn_matches = flann.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            print(f"Match error: {name1} vs {name2} - {e}")
            continue

        # 6.3 ratio test
        good_matches = []
        for knm in knn_matches:
            if len(knm) < 2:
                continue
            m, n = knm
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_inliers:
            continue

        # 6.4 Geometric check (Fundamental)
        pts1 = [keypoints_table[id1][m.queryIdx][:2] for m in good_matches]
        pts2 = [keypoints_table[id2][m.trainIdx][:2] for m in good_matches]
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, 0.99)
        if F is None or F.shape != (3, 3):
            continue

        inlier_mask = (mask.ravel() > 0)
        inlier_count = np.sum(inlier_mask)
        if inlier_count < min_inliers:
            continue

        inlier_matches = []
        for idx_m, m in enumerate(good_matches):
            if inlier_mask[idx_m]:
                inlier_matches.append([m.queryIdx, m.trainIdx])
        inlier_matches = np.array(inlier_matches, dtype=np.uint32)

        print(f" Pair ({name1}, {name2}): {inlier_count} inliers.")
        successful_matches += 1

        # 6.5 Write to DB
        pair_id = image_ids_to_pair_id(id1, id2)
        config = 2  # 2 = "F verified + E/H not computed"

        db.execute(
            """INSERT INTO two_view_geometries 
               (pair_id, rows, cols, data, config, F, E, H, qvec, tvec) 
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                pair_id,
                inlier_matches.shape[0],
                inlier_matches.shape[1],
                array_to_blob(inlier_matches),
                config,
                array_to_blob(F.astype(np.float64)),
                array_to_blob(np.eye(3, dtype=np.float64)),      # E (placeholder)
                array_to_blob(np.eye(3, dtype=np.float64)),      # H (placeholder)
                array_to_blob(np.array([1, 0, 0, 0], dtype=np.float64)),  # qvec
                array_to_blob(np.zeros(3, dtype=np.float64)),             # tvec
            )
        )

        db.execute(
            """INSERT INTO matches 
               (pair_id, rows, cols, data) 
               VALUES (?,?,?,?)""",
            (
                pair_id,
                inlier_matches.shape[0],
                inlier_matches.shape[1],
                array_to_blob(inlier_matches),
            )
        )

        # Print progress
        if idx % 100 == 0 or idx == total_pairs:
            print(f"Processed {idx}/{total_pairs} pairs.")

    # 7) Commit & close DB
    db.commit()
    db.close()

    print("Feature matching done.")
    print(f"total {successful_matches} valid pairs.")


def draw_keypoints_on_image(image_path, keypoints, save_path=None):
    """
     Draw keypoints on image.

    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert keypoints format
    if keypoints.shape[1] == 4:
        kp = [cv2.KeyPoint(pt[0], pt[1], pt[2], pt[3]) for pt in keypoints]
    elif keypoints.shape[1] == 2:
        kp = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in keypoints]
    else:
        print(" Unsupported keypoint dimension.")
        return

    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_with_kp)
        print(f"Keypoints image saved to: {save_path}")
    else:
        #  Show result
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


def extract_features_opencv_to_db(database_path, images_path, vis_path):
    """
    Extract SIFT via OpenCV & write into COLMAP DB.

    Args:
        database_path (str): Path to create/write the database.
        images_path (str): Folder containing all images.
        vis_path (str): Folder to save keypoint visualizations.

    Steps:
        1. Create & init DB.
        2. Extract SIFT for each image.
        3. Save keypoints & descriptors to DB.
    """
    print("# 1. Extract SIFT & write to DB")

    # If DB exists, remove it first
    if os.path.exists(database_path):
        print(f"warning: {database_path} DB exists. Overwriting.")
        os.remove(database_path)

    # 1) Init DB
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    # 2) Add a dummy camera (example)
    camera_id = 1
    model = 0  # SIMPLE_RADIAL = 0
    width, height = 3840, 2160
    params = np.array([1500.0, width / 2.0, height / 2.0], dtype=np.float64)
    db.add_camera(model, width, height, params, prior_focal_length=True, camera_id=camera_id)

    # 3) Init SIFT
    sift = cv2.SIFT_create(nfeatures=8192)

    # 4) Iterate images
    current_image_id = 1
    for fname in sorted(os.listdir(images_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        fpath = os.path.join(images_path, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Cannot read image: {fpath}")
            continue

        # 4.1  Extract SIFT
        keypts, desc = sift.detectAndCompute(img, None)
        if keypts is None or desc is None or len(keypts) == 0:
            print(f"image {fname} No features found.")
            continue

        print(f"Image: {fname},  #Keypoints: {len(keypts)}")

        # 4.2 Visualize keypoints
        keypoints_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypts], dtype=np.float32)
        draw_keypoints_on_image(fpath, keypoints_array, save_path=os.path.join(vis_path, fname))

        # 4.3 Add to images table
        db.add_image(name=fname, camera_id=camera_id, image_id=current_image_id)

        # 4.4 Add keypoints
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypts], dtype=np.float32)
        db.add_keypoints(image_id=current_image_id, keypoints=kp_array)

        # 4.5 Add descriptors (convert to uint8)
        desc_norm = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
        desc_float = desc / desc_norm
        desc_float *= 512.0
        desc_float = np.clip(desc_float, 0, 255)
        desc_uint8 = desc_float.astype(np.uint8)
        db.add_descriptors(image_id=current_image_id, descriptors=desc_uint8)

        current_image_id += 1

    db.commit()
    db.close()
    print(f"Features saved to {database_path}.")


def show_sparse_pointcloud(sparse_path):
    """
    Visualize sparse reconstruction from COLMAP binary.

    Args:
        sparse_path (str): Path to the sparse reconstruction folder.
    """
    print(f"Loading sparse point cloud: {sparse_path}")
    points3d_file = os.path.join(sparse_path, "points3D.bin")
    if not os.path.exists(points3d_file):
        print(f"error: {points3d_file}  Not found. Check reconstruction.")
        return

    points3D = rw.read_points3D_binary(points3d_file)
    print(f"Found {len(points3D)} points.")

    pts = [p.xyz for p in points3D.values()]
    if len(pts) == 0:
        print("Empty points. Cannot visualize.")
        return

    pts = np.array(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    print(f"点云范围 / Point cloud bound: min={pcd.get_min_bound()}, max={pcd.get_max_bound()}")
    print("开始可视化 / Starting visualization...")
    o3d.visualization.draw_geometries([pcd], window_name="Sparse Reconstruction")

def run_colmap_command(cmd, cwd=None):
    """
    Run COLMAP command and capture the output.
    """
    try:
        print(f"Run command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            cwd=cwd
        )
        print("Command output:")
        print(result.stdout)
        print("Command wrong output:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Command execution failed:")
        print("Command:", ' '.join(e.cmd))
        print("Returncode:", e.returncode)
        print("Standard output:", e.stdout)
        print("Error output:", e.stderr)
        raise

def set_colmap_path():
    """
    set COLMAP executable file。
    """
    # Here you need to modify it according to your actual position
    colmap_path = r"D:/AT_Master/Team_Project/colmap/bin/colmap.exe"
    if not os.path.isfile(colmap_path):
        print(f"error: can not find COLMAP.exe file '{colmap_path}'.please check the path.")
        return None
    print(f"COLMAPpath executable is set to: {colmap_path}\n")
    return colmap_path


def dense_reconstruction(colmap_path, images_path, sparse_path, dense_path):
    """
    Performs dense reconstruction steps, including image de-distortion, Patch Match Stereo, and Stereo Fusion.
    Use the COLMAP command line (subprocess) method.
    """
    print("# 5. Dense Reconstruction (subprocess)")

    # 5.1 Image De-distortion
    print("## 5.1 Image De-distortion")
    run_colmap_command([
        colmap_path, "image_undistorter",
        "--image_path", images_path,
        "--input_path", sparse_path,
        "--output_path", dense_path,
        "--output_type", "COLMAP",
        "--max_image_size", "5000"
    ])
    print("Image de-distortion complete.\n")

    # 5.2 Patch Match Stereo
    print("## 5.2 Patch Match Stereo")
    run_colmap_command([
        colmap_path, "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--log_to_stderr", "1",
        "--PatchMatchStereo.gpu_index", "0"
    ])
    print("Patch Match Stereo Complete.\n")

    # 5.3 Stereo Fusion
    print("## 5.3 Stereo Fusion")
    fused_ply = os.path.join(dense_path, "fused.ply")
    run_colmap_command([
        colmap_path, "stereo_fusion",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",  # or "photometric"
        "--output_path", fused_ply,
        "--StereoFusion.num_threads","16"
    ])
    print("Stereo Fusion 完成。\n")
    print(f"Final dense point cloud is output to: {fused_ply}")


def show_stereo_result(dense_ply_file):
    """
    show the dense result (fused.ply)。
    """
    if not os.path.exists(dense_ply_file):
        print(f"error: point cloud file '{dense_ply_file}' unexist.")
        return

    print(f"upload the point cloud file: {dense_ply_file}")
    pcd = o3d.io.read_point_cloud(dense_ply_file)
    print(f"The point cloud contains {len(pcd.points)} points to start the visualisation...")")

    # 初始化 GUI 系统
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # use O3DVisualizer 
    vis = o3d.visualization.O3DVisualizer("point cloud result")
    vis.add_geometry("point cloud", pcd)
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

    
def run_pipeline():
    """
    Main pipeline:
        1) Extract features 
        2) Match features 
        3) Sparse reconstruction
        4) Sparse reconstruction visualization
        5) Dense reconstruction
        6) Dense reconstruction visualisation
    """
    output_path = Path("workspace/")
    image_path = output_path / "images"
    database_path = output_path / "database.db"
    vis_path = output_path / "visualizations"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")

    if not image_path.exists():
        logging.error(" No input images.")
        raise FileNotFoundError("No input images folder.")

    #  Remove existing DB if any
    if database_path.exists():
        database_path.unlink()

    # 1) Extract features 
    extract_features_opencv_to_db(str(database_path), str(image_path), str(vis_path))

    # 2) Match features 
    match_features_opencv(str(database_path), str(vis_path))

    # 3) Sparse reconstruction
    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    recs = incremental_pipeline.run_incremental_sfm(database_path, image_path, sfm_path)
    for idx, rec in recs.items():
        logging.info(f"# {idx} {rec.summary()}")

    # 4) Sparse reconstruction visualization
    show_sparse_pointcloud(os.path.join(str(sfm_path), "0"))

    # 5) Dense reconstruction
    colmap_exe = set_colmap_path()
    if not colmap_exe:
        print("Unable to perform dense reconstruction: COLMAP executable not found.")
        return

    dense_dir = os.path.join(str(sfm_path), "dense")
    os.makedirs(dense_dir, exist_ok=True)

    dense_reconstruction(
        colmap_exe,
        images_path=str(image_path),
        sparse_path=os.path.join(str(sfm_path), "0"),
        dense_path=dense_dir
    )

    # 6) Dense reconstruction visualization
    dense_ply_file = os.path.join(str(dense_dir), "fused.ply") 
    show_stereo_result(dense_ply_file)


if __name__ == "__main__":
    run_pipeline()
