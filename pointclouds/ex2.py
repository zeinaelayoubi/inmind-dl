import open3d as o3d
import numpy as np




def drop_points(pcd, drop_percentage):
    
    num_points = len(pcd.points)
    num_to_drop = int(num_points * drop_percentage / 100)
    indices = np.random.choice(num_points, num_to_drop, replace=False)
    pcd = pcd.select_by_index(indices, invert=True)
    return pcd

def apply_jittering(pcd, jitter_strength):
    
    points = np.asarray(pcd.points)
    jitter = np.random.normal(0, jitter_strength, points.shape)
    jittered_points = points + jitter
    pcd.points = o3d.utility.Vector3dVector(jittered_points)
    return pcd

def add_noise(pcd, noise_stddev):
    
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, noise_stddev, points.shape)
    noisy_points = points + noise
    pcd.points = o3d.utility.Vector3dVector(noisy_points)
    return pcd

def slightly_change_color(pcd, color_change_strength):
   
    if not pcd.has_colors():
        raise ValueError("Point cloud does not have colors.")
    
    colors = np.asarray(pcd.colors)
    color_change = np.random.uniform(-color_change_strength, color_change_strength, colors.shape)
    new_colors = np.clip(colors + color_change, 0, 1)  # Ensure colors remain within [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return pcd

# Example usage
if __name__ == "__main__":
    # Load point cloud
    pcd = o3d.io.read_point_cloud("1.ply")
    o3d.visualization.draw_geometries([pcd])
    
    # Drop 10% of the points
    pcd = drop_points(pcd, 50)
    o3d.visualization.draw_geometries([pcd])
    
    # Apply jittering with a strength of 0.01
    pcd = apply_jittering(pcd, 10)
    o3d.visualization.draw_geometries([pcd])
    
    # Add Gaussian noise with a standard deviation of 0.01
    pcd = add_noise(pcd, 10)
    o3d.visualization.draw_geometries([pcd])
    
    # Slightly change colors with a strength of 0.1
    pcd = slightly_change_color(pcd, 1)
    o3d.visualization.draw_geometries([pcd])

