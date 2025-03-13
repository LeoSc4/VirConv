import numpy as np
import open3d as o3d
import os

def load_kitti_calib(calib_file):
    """
    Load KITTI calibration file
    Args:
        calib_file: Path to the calibration file
    Returns:qqqqqqqq
        dict: Calibration matrices
    """
    calib = {}
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in value.split()])

    # Reshape matrices
    calib['P0'] = calib['P0'].reshape(3, 4)
    calib['P1'] = calib['P1'].reshape(3, 4)
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['P3'] = calib['P3'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    # Create 4x4 transformation matrices for easier handling
    # Rectification matrix (rectifying camera coordinates)
    rect_4x4 = np.eye(4)
    rect_4x4[:3, :3] = calib['R0_rect']
    
    # Velodyne to camera transformation (includes both rotation and translation)
    velo_to_cam_4x4 = np.eye(4)
    velo_to_cam_4x4[:3, :4] = calib['Tr_velo_to_cam']
    
    # Complete transformation: velodyne -> unrectified camera -> rectified camera
    calib['velo_to_cam_rect'] = rect_4x4 @ velo_to_cam_4x4
    
    # Compute inverse transformation: rectified camera -> velodyne
    calib['cam_rect_to_velo'] = np.linalg.inv(calib['velo_to_cam_rect'])
    
    # For debugging, print the transformation matrices
    print("Velodyne to Camera Rectified:\n", calib['velo_to_cam_rect'])
    print("\nCamera Rectified to Velodyne:\n", calib['cam_rect_to_velo'])
    
    return calib

def load_kitti_labels(label_path):
    """
    Load KITTI label file and extract 3D bounding box information
    Args:
        label_path: Path to the KITTI label file
    Returns:
        bboxes: List of dictionaries containing bbox info for each object
    """
    bboxes = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:  # Basic validation
            continue
            
        # KITTI format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y [score]
        obj_type = parts[0]
        # Skip DontCare labels
        if obj_type == 'DontCare':
            continue

        # 2D bounding box parameters
        u1, v1, u2, v2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])


        # 3D bounding box parameters (in camera coordinate system)
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        rotation_y = float(parts[14])
        
        bboxes.append({
            'bbox_2d': [u1, v1, u2, v2],
            'type': obj_type,
            'dimensions': [l, h, w],  # KITTI order is l, h, w (length, height, width)
            'location': [x, y-0.93, z],    # Center of box                                           ############ Manual correction
            'rotation_y': rotation_y  # Rotation around Y-axis
        })
    
    return bboxes

def load_kitti_labels_raw(label_path):
    """
    Load KITTI label file and extract 3D bounding box information
    Args:
        label_path: Path to the KITTI label file
    Returns:
        bboxes: List of dictionaries containing bbox info for each object
    """
    bboxes = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:  # Basic validation
            continue
            
        # KITTI format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y [score]
        obj_type = parts[0]
        # Skip DontCare labels
        if obj_type == 'DontCare':
            continue

        # 2D bounding box parameters
        u1, v1, u2, v2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])


        # 3D bounding box parameters (in camera coordinate system)
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        rotation_y = float(parts[14])
        
        bboxes.append({
            'bbox_2d': [u1, v1, u2, v2],
            'type': obj_type,
            'dimensions': [l, h, w],  # KITTI order is l, h, w (length, height, width)
            'location': [x, y, z],    # Center of box                                           ############ Manual correction
            'rotation_y': rotation_y  # Rotation around Y-axis
        })
    
    return bboxes


def kitti_to_open3d_bbox(bbox_data, calib=None):
    """
    Convert KITTI format bounding box to Open3D OrientedBoundingBox
    Args:
        bbox_data: Dictionary with KITTI bbox info
        calib: Optional calibration dictionary to transform to velodyne coordinates
    Returns:
        o3d_bbox: Open3D OrientedBoundingBox object
    """
    # Extract parameters
    dimensions = bbox_data['dimensions']  # length(l), height(h), width(w)
    location = bbox_data['location']     # x, y, z (center)
    rotation_y = bbox_data['rotation_y'] + np.deg2rad(90) # rotation around y-axis                 ############ Manual correction
    obj_type = bbox_data['type']
    
    # Create rotation matrix from rotation_y
    # In KITTI, rotation_y is the rotation around y-axis in camera coordinates
    R_cam = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    
    # If calibration is provided, transform from camera to velodyne coordinates
    if calib is not None:
        # Create a 4x4 transformation matrix in camera coordinates for the bounding box
        box_cam = np.eye(4)
        box_cam[:3, :3] = R_cam
        box_cam[:3, 3] = location
        
        # Apply full camera-to-velo transformation (including both rotation and translation)
        box_velo = calib['cam_rect_to_velo'] @ box_cam
        
        # Extract the rotation matrix and translation vector from the resulting 4x4 matrix
        R_velo = box_velo[:3, :3]
        t_velo = box_velo[:3, 3]
        
        # Use the transformed rotation matrix and location
        R = R_velo
        location = t_velo
    else:
        # Use original values
        R = R_cam
    
    # Create bounding box
    bbox = o3d.geometry.OrientedBoundingBox()
    
    # Set box center
    bbox.center = np.array(location)
    
    # Set box rotation
    bbox.R = R
    
    # Set box extents (need to adjust for Open3D convention)
    # KITTI: length (x), height (y), width (z)
    # Open3D expects: width, height, length
    bbox.extent = np.array([dimensions[2], dimensions[1], dimensions[0]])
    
    # Set color based on object type
    color_map = {
        'Car': [1, 0, 0],       # Red
        'Pedestrian': [0, 1, 0], # Green
        'Cyclist': [0, 0, 1],    # Blue
        'Van': [1, 0.5, 0],      # Orange
        'Truck': [0.5, 0, 0.5]   # Purple
    }
    bbox.color = np.array(color_map.get(obj_type, [1, 1, 0]))  # Default yellow
    
    return bbox

def visualize_kitti_scene_with_calib(points_array, label_path=None, calib_path=None):
    """
    Visualize a point cloud array with 3D bounding boxes using calibration
    
    Args:
        points_array: NumPy array of shape (N, 4) with x, y, z, intensity values
        label_path: Path to the KITTI label file
        calib_path: Path to the KITTI calibration file
    """
    # Load calibration if provided
    calib = None
    if calib_path and os.path.exists(calib_path):
        calib = load_kitti_calib(calib_path)
        print("Calibration loaded successfully.")
        
        # Print the first few rows of the transformation matrices to verify
        print("\nVelodyne to Camera transformation matrix (first 3 rows):")
        print(calib['velo_to_cam_rect'][:3, :])
        
        print("\nCamera to Velodyne transformation matrix (first 3 rows):")
        print(calib['cam_rect_to_velo'][:3, :])
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array[:, :3])
    
    # Color the point cloud based on intensity if available
    if points_array.shape[1] >= 4:
        colors = np.zeros((len(points_array), 3))
        normalized_intensity = points_array[:, 3] / np.max(points_array[:, 3])
        colors[:, 0] = normalized_intensity #1.0  #normalized_intensity  # Map intensity to red channel
        colors[:, 1] = normalized_intensity #1.0  #normalized_intensity  # Map intensity to green channel
        colors[:, 2] = normalized_intensity #1.0  #normalized_intensity  # Map intensity to blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("KITTI Scene Visualization", width=1280, height=720)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Add bounding boxes if label file is provided
    if label_path and os.path.exists(label_path):
        bboxes = load_kitti_labels(label_path)
        print(f"Loaded {len(bboxes)} bounding boxes from label file.")
        
        for i, bbox_data in enumerate(bboxes):
            print(f"\nBox {i+1}: {bbox_data['type']}")
            print(f"  Camera coordinates: center={bbox_data['location']}, rotation_y={bbox_data['rotation_y']}")
            
            # Create Open3D bounding box with calibration
            o3d_bbox = kitti_to_open3d_bbox(bbox_data, calib)
            
            print(f"  Velodyne coordinates: center={o3d_bbox.center}")
            vis.add_geometry(o3d_bbox)
            
            # Add text labels using small spheres
            label_pos = o3d_bbox.center + np.array([0, o3d_bbox.extent[1]/2 + 0.3, 0])
            label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            label_sphere.paint_uniform_color(o3d_bbox.color)
            label_sphere.translate(label_pos)
            vis.add_geometry(label_sphere)
    
    # Set visualization parameters
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.set_lookat([0, 0, 0])
    
    # Run visualization
    vis.run()
    vis.destroy_window()

# Example usage
def main():
    # Replace this with your actual preprocessed point cloud data
    
    # Example for testing (creating dummy data)
    # points_array = np.load("data/kitti/training/velodyne_depth/000008.npy") # original loading 

    # with SampleKITTIdataset
    points_array = np.fromfile('data/kitti/training/velodyne/000001.bin', dtype=np.float32).reshape(-1, 4)

    # points_array = np.load("data/kitti/training/velodyne/000008.npy") 


    
    # Path to your KITTI files
    label_path = 'data/kitti/training/label_2/000008.txt'
    # calib_path = '000008_calib.txt'  # Using the provided calibration file
    calib_path = '/workspace/data/kitti/training/calib/000008.txt'  # Using the provided calibration file
    
    # Visualize the point cloud with bounding boxes and calibration
    visualize_kitti_scene_with_calib(points_array, label_path, calib_path)

if __name__ == "__main__":
    main()