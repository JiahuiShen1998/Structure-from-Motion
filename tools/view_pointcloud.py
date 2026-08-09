"""Open a point cloud in an Open3D window.

Convenience viewer for the pipeline's outputs: the fused dense cloud
(`workspace/sfm/dense/fused.ply`) or any other PLY. For the *sparse* result, stage 05
reads `points3D.bin` directly via COLMAP's `read_write_model`; this script only handles
formats Open3D can open.

Input   a PLY (or any format `open3d.io.read_point_cloud` accepts)
Output  an interactive window; nothing is written

Needs a display. On a headless machine it will fail at window creation, not at load --
use --stats to get the point count and bounding box without opening a window.

Original name: show_points.py, which had the path hardcoded.
"""

import argparse
import os

import open3d as o3d


def load_point_cloud(path):
    """Read a point cloud and fail loudly if it is missing or empty.

    Open3D returns an empty cloud rather than raising when a file is unreadable or
    contains no vertices, which otherwise surfaces as a blank window.
    """
    if not os.path.exists(path):
        raise SystemExit(f"No such file: {path}")

    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise SystemExit(
            f"{path} loaded but contains no points. If this is a COLMAP dense result, "
            "check that stereo_fusion actually completed."
        )
    return pcd


def describe(pcd):
    """Print point count and bounding box -- enough to tell a real cloud from a stub."""
    print(f"[INFO] {len(pcd.points)} points")
    print(f"[INFO] bounds min={pcd.get_min_bound()} max={pcd.get_max_bound()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="View a point cloud produced by the pipeline."
    )
    parser.add_argument("ply", help="Path to a .ply point cloud")
    parser.add_argument("--stats", action="store_true",
                        help="Print point count and bounds, then exit without opening "
                             "a window (works headless)")
    parser.add_argument("--window-name", default="Point cloud",
                        help="Title of the viewer window")
    return parser.parse_args()


def main():
    args = parse_args()
    pcd = load_point_cloud(args.ply)
    describe(pcd)
    if args.stats:
        return
    o3d.visualization.draw_geometries([pcd], window_name=args.window_name)


if __name__ == "__main__":
    main()
