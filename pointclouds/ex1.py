import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("1.ply")

# Compute the minimum and maximum of x, y, z
min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()
print(f"Min bound: {min_bound}")
print(f"Max bound: {max_bound}")


center = pcd.get_center()
range_x = max_bound[0] - min_bound[0]
range_y = max_bound[1] - min_bound[1]
range_z = max_bound[2] - min_bound[2]

print(f"Range in x: {range_x}")
print(f"Range in y: {range_y}")
print(f"Range in z: {range_z}")

print("Displaying original point cloud...")
o3d.visualization.draw_geometries([pcd])


distances = np.linalg.norm(np.asarray(pcd.points) - center, axis=1)


distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

# Apply color gradient (e.g., from red to blue)
colors = np.zeros((len(distances), 3))
colors[:, 0] = 1.0 - distances_normalized  # Red
colors[:, 2] = distances_normalized        # Blue 

pcd.colors = o3d.utility.Vector3dVector(colors)

# display the colored point cloud
print("Displaying point cloud with color gradient...")
o3d.visualization.draw_geometries([pcd])

############################################


plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)


print("Plane model: ", plane_model)

# Color the inliers (plane) and outliers differently for visualization
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for the plane
outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for the rest


o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  window_name="Segmented Plane Visualization",
                                  point_show_normal=True)