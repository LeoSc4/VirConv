from tools.visual_utils import open3d_vis_utils as VisOpen3D
import numpy as np 

import open3d as o3d

# vis_utils from LS

def load_kitti_calib(calib_file):
    """
    Load KITTI calibration file
    Args:
        calib_file: Path to the calibration file
    Returns:
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
            'dimensions': [h, w, l],         
            'location': [x, y, z],    # Center of box                              
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
    dimensions = bbox_data['dimensions']  # h,w,l from labels_raw
    location = bbox_data['location']     # x, y, z (center)
    rotation_y = bbox_data['rotation_y'] # rotation around y-axis                 
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

        # Divide the z value by 2 to match the visualization of bb by using centeroid as location point
        t_velo[2] /= 2

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
    # KITTI: height, width, length
    # Open3D expects: width, height, length
    # bbox.extent = np.array([dimensions[2], dimensions[0], dimensions[1]])          #width/ length depending on definition in 
    bbox.extent = np.array([dimensions[1], dimensions[0], dimensions[2]])          #width/ length depending on definition in 
    

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


def get_rotation_matrices_from_y(rotation_y):
    """
    Create 3x3 rotation matrices from an array of rotation angles around the y-axis.

    Args:
        rotation_y (np.ndarray): Array of rotation angles in radians.

    Returns:
        np.ndarray: Array of 3x3 rotation matrices.
    """
    if isinstance(rotation_y, np.ndarray):   #if rotation_y angles for multiple boxes are provided
        cos_theta = np.cos(rotation_y)
        sin_theta = np.sin(rotation_y)
        
        rotation_matrices = np.zeros((rotation_y.shape[0], 3, 3))
        rotation_matrices[:, 0, 0] = cos_theta
        rotation_matrices[:, 0, 2] = sin_theta
        rotation_matrices[:, 1, 1] = 1
        rotation_matrices[:, 2, 0] = -sin_theta
        rotation_matrices[:, 2, 2] = cos_theta
        
        return rotation_matrices
    
    else:                                   #if only one rotation_y angle is provided
        cos_theta = np.cos(rotation_y)
        sin_theta = np.sin(rotation_y)
        
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        
        return rotation_matrix
    

def extract_rotation_y_from_matrices(matrices):
    """
    Extract the rotation_y angles from an array of 4x4 rotation matrices.

    Args:
        matrices (np.ndarray): Array of 4x4 rotation matrices with shape (N, 4, 4).

    Returns:
        np.ndarray: Array of rotation_y angles in radians with shape (N,) or a single rotation_y angle.
    """
    if matrices.ndim == 3:  # if multiple matrices == BBs are provided
        return np.arctan2(matrices[:, 0, 2], matrices[:, 0, 0])
    elif matrices.ndim ==2:              # if only one matrix == BB is provided
        return np.arctan2(matrices[2,0], matrices[0,0])
    else:
        return ValueError("Input must be a 4x4 matrix or an array of 4x4 matrices")
    

def visualize_pc_bbox_results(selected_frame, det_annos, gt_boxes_in_camera_cf, gt_boxes_in_velo_cf, visualization_frame_data_dict): 
    print("----------------- VISUALIZATION -----------------")    
    print("Visualized frame: ", det_annos[selected_frame]['frame_id'])
    print("Visualized prediction bounding boxes: ", det_annos[selected_frame]['name'])
    
    # Attention: Using the pred_dicts will show the all predicted BB (-> WBF (advanced NMS) is not applied)
    # If you want to show the WBF applied BB, you need to use the det_annos[selected_frame]['boxes_lidar'] instead of the pred_dicts
    # VisOpen3D.draw_scenes(
    #     points=visualization_frame_data_dict['points'][:, :3],     
    #     gt_boxes=None,
    #     ref_boxes=prediction_dicts[selected_frame][0]['pred_boxes'],           
    #     ref_labels=prediction_dicts[selected_frame][0]['pred_labels'], 
    #     ref_scores=None     # optional: pred_dicts[0]['pred_scores']     # optional
    #     )

    print("gt_boxes_in_velodyne_camera: \n", gt_boxes_in_camera_cf)
    print("gt_boxes_in_velodyne_cf: \n", gt_boxes_in_velo_cf)
    # compare_arrays(gt_boxes_in_velodyne_cf, gt_boxes_in_camera_cf)

    # print("----- VIZ in native coordinate system ------")    
    # VisOpen3D.draw_scenes(
    #     points=visualization_frame_data_dict['points'][:, :3],      #input points as part of frame_dict
    #     gt_boxes=gt_boxes_in_camera_cf,
    #     ref_boxes=None, #det_annos[selected_frame]['boxes_lidar'],            # # ref_boxes=pred_dicts[2]['pred_boxes'] -> könnte man auch aus den pred_dicts holen, wenn man die Berechnung der boxes_lidar verfolgt
    #     ref_labels=None,                                            #det_annos[selected_frame]['name'],     # optional: pred_dicts[0]['pred_labels'],    
    #     ref_scores=None                                         # optional: pred_dicts[0]['pred_scores']     # optional
    #     )
    
    print("----- VIZ in Velodyne coordinate system system ------")    
    VisOpen3D.draw_scenes(
        points=visualization_frame_data_dict['points'][:, :3],      
        gt_boxes=gt_boxes_in_velo_cf,
        ref_boxes=None, # det_annos[selected_frame]['boxes_lidar'],            
        ref_labels=None,                                            
        ref_scores=None                                         
        )


def save_point_cloud_as_pcd(points, filename):
    """
    Saves the point cloud as .pcd file. 
    
    Args:
        points (numpy.ndarray): Point cloud as Nx3 or Nx4 array.
        filename (str): filename of the .pcd
    """
    if points.shape[1] != 4: 
        print("shape of the input points: ", points.shape)

    point_cloud = o3d.geometry.PointCloud()
    
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Optional: Set colors if existens 
    # if points.shape[1] == 4:
        # point_cloud.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
    
    # Save the point cloud as .pcd file
    o3d.io.write_point_cloud(filename, point_cloud)



def compare_arrays(array1, array2):
    """
    Compare two arrays and print the differences in shape, values, and data types.

    Args:
        array1 (np.ndarray): First array to compare.
        array2 (np.ndarray): Second array to compare.
    """
    # Compare shapes
    if array1.shape != array2.shape:
        print("Shapes are different:")
        print("Shape of array1:", array1.shape)
        print("Shape of array2:", array2.shape)
    else:
        print("Shapes are the same:", array1.shape)

    # Compare data types
    if array1.dtype != array2.dtype:
        print("Data types are different:")
        print("Data type of array1:", array1.dtype)
        print("Data type of array2:", array2.dtype)
    else:
        print("Data types are the same:", array1.dtype)

    # Compare values
    # if not np.array_equal(array1, array2):
    #     print("Values are different:")
    #     diff = np.abs(array1 - array2)
    #     print("Difference array:\n", diff)
    # else:
    #     print("Values are the same")

def cl_prints(batch_dict, pred_dicts, det_annos, i): 
    print("--------#### New Frame #### -----------")
    print("--------- BATCH_DICT --------------")

    #print the number of the first used element of the batch dict 
    print("Current iteration: ", i)
    print("Currently used batch element ", batch_dict['frame_id'])
    
    print("--------- PREDICTION_DICT ---------")
    # print the name of the first prediction
    print("Predicted label: ", pred_dicts[0]['pred_labels']) # print the pred_labels of the prediction
    print("Predicted bounding box: \n", pred_dicts[0]['pred_boxes']) # print the pred_boxes of the prediction
    print("Predicted score = confidence: ", pred_dicts[0]['pred_scores']) # print the pred_scores of the prediction
    
    print("-------------- ANNOS --------------")
    # Print the class of the first element 
    print("Class of the first element: ", det_annos[i]['name'])
    # Print the score of the first element
    print("Score of the first element: ", det_annos[i]['score'])

    #2D Bounding Box
    print("--------- 2D Bounding Box ---------")

    #insert a statement that is only executed if the line is successful 
    if len(det_annos[i]['bbox']) > 0:       # if something is detected
        #print the 2D Bounding Box in the image
        print("2D BB in the image: \n", det_annos[i]['bbox'])
        print("2D BB for the current prediction: \n", det_annos[0]['bbox'][0])
        print("-- Left pixel: ", det_annos[i]['bbox'][0][0])
        print("-- Top pixel: ", det_annos[i]['bbox'][0][1])
        print("-- Right pixel: ", det_annos[i]['bbox'][0][2])
        print("-- Bottom pixel: ", det_annos[i]['bbox'][0][3])
    else:
        print("2D BB in the image: Not available")

    # 3D Bounding Box
    if len(det_annos[i]['dimensions']) > 0:
        print("-------- 3D Bounding Box ---------")
        print("Dimensions of the first 3D BB in frame in meters:")
        print("--Height: ", det_annos[i]['dimensions'][0][0]) # last [0] due to the possibility for multiple detected BB with Height, Width, Length each
        print("--Width: ", det_annos[i]['dimensions'][0][1])
        print("--Length: ", det_annos[i]['dimensions'][0][2])

        ## Print the location of the 3D Bounding Box
        print("Location of the 3D BB in camera coordinates in meters: \n", det_annos[i]['location'])
        ## Print the rotation around y-axis in camera coordinates of the 3D Bounding Box
        print("Rotation around y-axis in camera coordinates of the 3D BB in radians: ", det_annos[i]['rotation_y'])

        print("Score for eval: ", det_annos[i]['score'])
    else:
        print("3D BB in frame: Not available")