import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_camera_pose_omnv_world(omnv_def_cam_pose_omnv_world_path):
    # get omniverse default convention camera pose in omniverse world coordinate frame
    with open(omnv_def_cam_pose_omnv_world_path) as f:
        omnv_def_cam_poses_omnv_world = json.load(f)

    omnv_def_cam_positions = omnv_def_cam_poses_omnv_world['camera_positions']
    omnv_def_cam_rotations = omnv_def_cam_poses_omnv_world['camera_rotations']

    cam_transform_matrices = []

    for frame in range(len(omnv_def_cam_positions)):
        position = omnv_def_cam_positions[frame]
        rotation_deg = omnv_def_cam_rotations[frame]  # [pitch, yaw, roll] in degrees

        # Convert from degree to radian: Assumed order = roll, pitch, yaw (== xyz order)
        rotation = R.from_euler('xyz', np.radians(rotation_deg), degrees=False)
        rot_matrix = rotation.as_matrix()

        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = position

        # Add transformation of the kitti camera (y down, z forward) to the omniverse def camera (y up, z backwards)
        Tr_kitti_cam_to_omni_world = np.eye(4)
        Tr_kitti_cam_to_omni_world[1, 1] = -1
        Tr_kitti_cam_to_omni_world[2, 2] = -1

        # Get the transformation matrix from KITTI camera to Omniverse world coordinates
        Tr_kitti_cam_to_omni_world = T @ Tr_kitti_cam_to_omni_world

        frame_id = f"{frame:06d}"  #add the frame id to the dictionary. Assumption: frame id is the index of the frame in the list (dataset generation corresponds to the defined camera trajectory)

        cam_transform_matrices.append({
            "Tr_kitti_cam_to_omni_world": Tr_kitti_cam_to_omni_world, 
            "frame_id": frame_id
        })    

        print(f"Frame {frame_id} - Camera Transformation Matrix:\n{T}\n")

    return cam_transform_matrices

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

def reconstruct_bbox_pose_omnv_world_standalone(cam_transform_matrices, pred_bboxes_poses_kitti_cam, current_frame_id):
    # Transforms each predicted bounding box pose PER FRAME from camera coordinates to Omniverse world coordinates.

    bbox_poses_omnv_world = []

    for frame in cam_transform_matrices:
        if frame['frame_id'] == current_frame_id:
            # Get the cam_T for the current frame 
            cam_T = frame['Tr_kitti_cam_to_omni_world']

            for i in range(len(pred_bboxes_poses_kitti_cam)):     # BBOXES must be available for specific frame
                bbox_T = pred_bboxes_poses_kitti_cam[i]['pred_bbox_pose_kitti_cam']

                bbox_world = cam_T @ bbox_T

                bbox_poses_omnv_world.append(bbox_world)

                # Print the transformed bounding box poses
                print(f"Bbox pose in omniverse world coordinates: \n{bbox_world}")

    return bbox_poses_omnv_world


def reconstruct_bbox_pose_omnv_world(cam_transform_matrices, pred_bbox_center_kitti_cam, current_frame_id):
    # Transforms each predicted bounding box pose PER FRAME from camera coordinates to Omniverse world coordinates.

    bbox_poses_omnv_world = []

    for frame in cam_transform_matrices:
        if frame['frame_id'] == current_frame_id:
            # Get the cam_T for the current frame 
            cam_T = frame['Tr_kitti_cam_to_omni_world'] # 4x4 

            bbox_center_omnv_world = cam_T @ pred_bbox_center_kitti_cam

            # Print the transformed bounding box poses
            print(f"Bbox pose in omniverse world coordinates: \n{bbox_center_omnv_world}")

    # Error check: If bbox_center_omnv_world is empty, print the current_frame_id
    if bbox_center_omnv_world is None:
        print(f"Error: bbox_center_omnv_world is empty for frame_id: {current_frame_id}")
        return None

    return bbox_center_omnv_world

def map_class_name(name_kitti): 
    # Map the class name from KITTI to our application
    class_mapping = {
        'Car': 'Trolley_RU2',
        # Add more mappings as needed

    }
    return class_mapping.get(name_kitti, 'Unknown')  # Default to 'Unknown' if not found


def get_asset_path_omnv(class_name):
    
    asset_path_mapping = {
        'Trolley_RU2': '/home/leo/workspace/Omniverse/OwnAssets/Trolley_RU2/RU2_dolly.usdc',
        # Add more mappings as needed
    }

    return asset_path_mapping.get(class_name, 'Unknown')  # Default to 'Unknown' if not found


if __name__ == "__main__":
    # load kitti camera pose in omniverse world coordinate frame from json file 
    # Path:'/workspace/tools/workspace/camera_poses.json'

    omnv_def_cam_pose_omnv_world_path = '/workspace/tools/workspace/camera_poses.json'
    pred_bbox_pose_kitti_cam_path = '/workspace/tools/workspace/mocked_pred_boxes.csv'

    cam_transform_matrices = get_camera_pose_omnv_world(omnv_def_cam_pose_omnv_world_path)


    ####### Currently mocked data and only for one frame #######
    pred_bbox_pose_kitti_cam = get_bbox_pose_kitti_cam(pred_bbox_pose_kitti_cam_path)

    # frame id: format = 000000
    mocked_frame_ids = ['000000', '000001', '000002', '000003']

    bbox_poses_omnv_world_for_all_frames = []

    # Iterate over the mocked frame ids and reconstruct the bounding box poses
    for frame in range(len(mocked_frame_ids)):
        current_frame_id = mocked_frame_ids[frame]
        bbox_pose_omnv_world = reconstruct_bbox_pose_omnv_world_standalone(cam_transform_matrices, pred_bbox_pose_kitti_cam, current_frame_id)
        bbox_poses_omnv_world_for_all_frames.append(bbox_pose_omnv_world)
        
        print(f"Frame {mocked_frame_ids[frame]} - Bbox poses in omniverse world coordinates: \n{bbox_pose_omnv_world}\n")