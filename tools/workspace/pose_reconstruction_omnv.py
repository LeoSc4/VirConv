import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def xform_matrix_xyz(rx_deg, ry_deg, rz_deg, t):
    """Omniverse-conform 4×4-Matrix from euler angles (Grad) + Translation"""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    # Omniverse uses row-major matrices, so we construct the rotation matrix accordingly
    R = np.array([
        [cy*cz,  sx*sy*cz - cx*sz,  sx*sz + cx*sy*cz],
        [cy*sz,  sx*sy*sz + cx*cz, -sx*cz + cx*sy*sz],
        [-sy,    sx*cy,             cx*cy           ]
    ])

    # Create the 4x4 transformation matrix  
    ## ATTENTION: Not ROW-MAJOR Matrix, but COLUMN-MAJOR Matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t          # t = (tx, ty, tz)

    return T

def get_USD_cam_pose_in_WORLD(USD_cam_in_world_path):
    # get omniverse default convention camera pose in omniverse world coordinate frame
    with open(USD_cam_in_world_path) as f:
        USD_cam_poses_omnv_world = json.load(f)

    USD_cam_positions = USD_cam_poses_omnv_world['camera_positions']
    USD_cam_rotations = USD_cam_poses_omnv_world['camera_rotations']

    # NOTE: ROTATIONS Values are in degrees and in the order [pitch, yaw, roll] AND as a Global Rotation with WORLD fixed axes (Extrinsics sequence) 
        # A Local Rotation Matrix must be created 

    cam_extrinsics_matrices = []

    for frame in range(len(USD_cam_positions)):
        position = USD_cam_positions[frame]
        rotation_deg = USD_cam_rotations[frame]  # [pitch, yaw, roll] in degrees
        rx_deg, ry_deg, rz_deg = rotation_deg  

        print(f"DEBUG: Frame {frame} - Camera Rotation (degrees): {rotation_deg}")

        # Get column-major T-Mat for camera extrinsics (pose in WORLD)
        CAM_extrinsics_T_mat = xform_matrix_xyz(rx_deg, ry_deg, rz_deg, position)

        print(f"DEBUG: Frame {frame} - Camera Extrinsics Matrix:\n{CAM_extrinsics_T_mat}")

        frame_id = f"{frame:06d}"  #add the frame id to the dictionary. Assumption: frame id is the index of the frame in the list (dataset generation corresponds to the defined camera trajectory)

        cam_extrinsics_matrices.append({
            "frame_id": frame_id,
            "T_USD_CAM_extrinsics": CAM_extrinsics_T_mat 
        })    

    return cam_extrinsics_matrices

def get_bbox_pose_kitti_cam(pred_bbox_pose_kitti_cam_path):    
    # load data from csv file
    # content: name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_cam, y_cam, z_cam, rotation_y, score
   
    # Output:
        # BBox pose in the camera coordinate frame as format to multiply with transform matrix

    with open(pred_bbox_pose_kitti_cam_path, 'r') as f:
        bbox_lines_with_header = f.readlines()
        bbox_lines = [line.strip().split(',') for line in bbox_lines_with_header[1:]]  # Skip the header

    # Extract the bbbox_poses in the camera coordinate frame
    pred_bboxes_poses_kitti_cam = []
    
    for bbox in bbox_lines:
        name = bbox[0]
        x, y, z = float(bbox[11]), float(bbox[12]), float(bbox[13])
        rotation_y = float(bbox[14]) # yaw in radians (KITTI format)
        rot_matrix = R.from_euler('y', rotation_y, degrees=False).as_matrix()
        
        # Create a 4x4 transformation matrix [R| t]
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [x, y, z]

        pred_bboxes_poses_kitti_cam.append({
            'name': name,
            'pred_bbox_pose_kitti_cam': T
            })

        print(f"Bbox pose: \n{T}")


    return pred_bboxes_poses_kitti_cam

def map_class_name(name_kitti): 
    # Map the class name from KITTI to our application
    class_mapping = {
        'Car': 'Trolley_RU2',
        # Add more mappings as needed

    }
    return class_mapping.get(name_kitti, 'Unknown')  # Default to 'Unknown' if not found


def get_asset_path_omnv(class_name):
    asset_path_mapping = {
        'Trolley_RU2': '/home/simulation/workspace/Assets/Trolley_RU2/Ru2_Dolly.usd'    #'/home/leo/workspace/Omniverse/OwnAssets/Trolley_RU2/RU2_dolly.usdc',
        # Add more mappings as needed
        }

    return asset_path_mapping.get(class_name, 'Unknown')  # Default to 'Unknown' if not found

