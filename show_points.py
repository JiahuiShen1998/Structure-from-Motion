import open3d as o3d

# Load PLY file
ply_file = r"./room_right.ply"
point_cloud = o3d.io.read_point_cloud(ply_file)

# Print basic info
print(point_cloud)

# Visualize
o3d.visualization.draw_geometries([point_cloud])
