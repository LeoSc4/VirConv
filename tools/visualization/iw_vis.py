import numpy as np
import open3d as o3d


from tools.visual_utils.vis_utils_ls import load_kitti_calib
from tools.visual_utils.vis_utils_ls import save_point_cloud_as_pcd


def load_kitti_labels_in_velo(label_path, calib_path): 
    """
    Load KITTI label file and 2D and 3D BBox information in Velodyne CF
    Args:
        label_path: Path to the KITTI label file
        calib_path: Path to the KITTI calibration file
    Returns:
        bboxes: List of dictionaries containing bbox info for each object (in Velodyne CF) 
    """
    bboxes_velo = []

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

        # 3D bounding box parameters
        center = np.array([float(parts[11]), float(parts[12]), float(parts[13]), 1.0]) #x, y, z         
        rotation_y = float(parts[14])  # Rotation around Y-axis
        
        
        # Alternative: Direct transformation of 3D bbox to Velodyne coordinate frame 
        size = [float(parts[9]), float(parts[10]), float(parts[8])]  # width, length, height
        
        calib = load_kitti_calib(calib_path)
        center_velo = calib['cam_rect_to_velo'] @ center                        
        center_velo = [center_velo[0], center_velo[1], center_velo[2]]      # /2 -> changed starting from dataset 8: using the center of the min and max x,y,z from the 3D bounding box instead of assuming the center is at 0
        
        rotation_y_velo = np.pi - rotation_y  # Convert from camera frame to lidar frame

        # Store the bbox information
        bboxes_velo.append({
            'type': obj_type,
            'bbox_2d': [u1, v1, u2, v2],
            'dimensions': size, 
            'location': center_velo, 
            'rotation_y': rotation_y_velo,
            })
        
    return bboxes_velo


def load_kitti_labels_in_cam(label_path): 
    """
    Load KITTI label file and 2D and 3D BBox information in KITTI camera CF
    Args:
        label_path: Path to the KITTI label file
    Returns:
        bboxes: List of dictionaries containing bbox info for each object (in Velodyne CF) 
    """
    bboxes_cam = []

    # get current frame_id from label_path 
    # typical path= '/workspace/data/kitti/training/label_2/000000.txt'
    frame_id = label_path.split('/')[-1].split('.')[0].zfill(6) # Extract the frame ID from the file name

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

        # 3D bounding box parameters
        center = np.array([float(parts[11]), float(parts[12]), float(parts[13]), 1.0]) #x, y, z         
        rotation_y = float(parts[14])  # Rotation around Y-axis
        
        size = [float(parts[9]), float(parts[10]), float(parts[8])]  # width, length, height
        
        center_cam = [center[0], center[1], center[2]]      # /2 -> changed starting from dataset 8: using the center of the min and max x,y,z from the 3D bounding box instead of assuming the center is at 0
        
        # Store the bbox information
        bboxes_cam.append({
            'type': obj_type,
            'bbox_2d': [u1, v1, u2, v2],
            'dimensions': size, 
            'location': center_cam, 
            'rotation_y': rotation_y,
            'frame_id': frame_id,  
            })
        
    return bboxes_cam

def create_bounding_box(label): 
    dimensions = np.array(label['dimensions'])
    location = np.array(label['location'])
    rotation_y = label['rotation_y']
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = location
    bbox.extent = dimensions
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rotation_y))
    oriented_bbox = bbox.rotate(rotation_matrix, center=bbox.center)
    oriented_bbox.color = [0, 1, 0] # default is green
    return oriented_bbox


def visualize_scene(points, gt_labels=None, predicted_bboxes=None, selected_frame=None): 
    """"
    Visualize the scene with Open3D. Labels must be in the velodyne cf (of points) before creating bounding boxes.
    """

    filter_point_cloud_by_range = False

    if filter_point_cloud_by_range:
        point_cloud_range = [0, -16, -3, 16, 16, 1]        #[0, -40, -3, 70.4, 40, 1] 

        mask = (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) \
        & (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4]) \
        & (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] <= point_cloud_range[5])
        points = points[mask]


    # Setup Open3D point cloud instance
    points_pcd = o3d.geometry.PointCloud()
  
    points_pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])  # use only x,y, z 

    # save_point_cloud_as_pcd(points, "/workspace/data/dataset_frame_check.pcd")
  
    intensity = points[:, 3]
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-5)  # normalize to [0,1]
    colors = np.stack([intensity]*3, axis=-1)  # grayscale RGB
    # colors = np.zeros((points.shape[0],3))    #ensure that color has the same length as points = [N,3]
    # colors[:, 0], colors[:, 1], colors[:, 2] = intensity, intensity, intensity  # set intensity as color for each point and each channel
    points_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Draw origin / coordinate frame into 3D 
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    window_title = f"Visualizing Frame: {selected_frame}" if selected_frame is not None else "Visualizing Scene"

    # Visualize Point Cloud 
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)
    vis.add_geometry(points_pcd)
    vis.add_geometry(coordinate_frame)

    if gt_labels is not None: 
        for label in gt_labels: 
            # Add gt_labels in green
            bbox = create_bounding_box(label)
            bbox.color = [0, 1, 0]
            vis.add_geometry(bbox)

    if predicted_bboxes is not None:
        for label in predicted_bboxes: 
            # Add predicted labels in red
            bbox = create_bounding_box(label)
            bbox.color = [1, 0, 0]
            vis.add_geometry(bbox)

    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = 2.0 
 

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 2])        
    ctr.set_front([-1, 0, 0])         # direction along x axis
    ctr.set_up([0, 0, 1])            # Z nach oben
    ctr.set_zoom(0.05)  
    
    vis.run()
    vis.destroy_window()

def main(points_path, calib_path=None, labels_path=None,iw_custom_data=False, kitti_reference_data=False):

    print("Current evaluation frame: ", points_path)
    # Load the point cloud 
    if iw_custom_data:
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4) # -1 to infer the length of the array
    
    elif kitti_reference_data:
        points = np.load(points_path) 

    if calib_path is not None and labels_path is not None: 
        gt_labels = load_kitti_labels_in_velo(labels_path, calib_path)
    
    visualize_scene(points, gt_labels=gt_labels)


if __name__ == '__main__':

    iw_custom_data = True 
    kitti_reference_data = False

    frame_idx = 8 #5238  #0

    # Filter point cloud by range enabled?

    if iw_custom_data:
        # points_path = f'/workspace/data/KITTI_000008_frame/training/velodyne/{str(frame_idx).zfill(6)}.bin'
        # calib_path = f'/workspace/data/KITTI_000008_frame/training/calib/{str(frame_idx).zfill(6)}.txt'
        # labels_path = f'/workspace/data/KITTI_000008_frame/training/label_2/{str(frame_idx).zfill(6)}.txt'

        points_path = f"/workspace/data/iw_dataset8-2_sample/training/velodyne/{str(frame_idx).zfill(6)}.bin"      #f"/workspace/data/kitti/training/velodyne/{str(frame_idx).zfill(6)}.bin"
        calib_path =  f"/workspace/data/iw_dataset8-2_sample/training/calib/{str(frame_idx).zfill(6)}.txt"      #f'/workspace/data/kitti/training/calib/{str(frame_idx).zfill(6)}.txt'
        labels_path = f"/workspace/data/iw_dataset8-2_sample/training/label_2/{str(frame_idx).zfill(6)}.txt"     #f'/workspace/data/kitti/training/label_2/{str(frame_idx).zfill(6)}.txt'

    if kitti_reference_data: 
        # points_path = f'/workspace/data/Reference_Subset_One/data/kitti/training/velodyne_depth/{str(frame_idx).zfill(6)}.npy'
        # calib_path = f'/workspace/data/Reference_Subset_One/data/kitti/training/calib/{str(frame_idx).zfill(6)}.txt'
        # labels_path = f'/workspace/data/Reference_Subset_One/data/kitti/training/label_2/{str(frame_idx).zfill(6)}.txt'
        points_path = f'/workspace/data/KITTI_000008_frame/training/velodyne_depth/{str(frame_idx).zfill(6)}.npy'
        calib_path = f'/workspace/data/KITTI_000008_frame/training/calib/{str(frame_idx).zfill(6)}.txt'
        labels_path = f'/workspace/data/KITTI_000008_frame/training/label_2/{str(frame_idx).zfill(6)}.txt'


    main(points_path, calib_path, labels_path, iw_custom_data, kitti_reference_data)  


