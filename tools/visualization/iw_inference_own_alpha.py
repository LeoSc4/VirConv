import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from tools.eval_utils import eval_utils
from pcdet.utils import common_utils

from visual_utils import open3d_vis_utils as VisOpen3D
from pcdet.utils import box_utils
from pcdet.datasets.kitti import kitti_dataset_mm
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common

import pickle

import datetime

import warnings
warnings.filterwarnings("ignore")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="cfgs/models/kitti/VirConv-T.yaml", help='specify the config for inference')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for inference')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)


    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

#### Needed ?
def load_kitti_calib(calib_file):
    """
    load projection matrix: https://github.com/AI-liu/Complex-YOLO/blob/master/utils.py#L197
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def get_rotation_matrix_from_y(rotation_y):
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

def main():

    print("Current working directory: ", Path.cwd())
    
    args, cfg = parse_config()
 
    log_dir = 'workspace/inference' 
    #create a log dir if not already available 
    log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('log_inference_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    #Logger is required for .load_params_from_file function
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    model_file_path = '../output/pretrained_models/VirConv-T-Paper.pth'

    # Build the dataloader for inference
    inference_dataset, inference_dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,    #dataset config defined in .yaml of model -> dataset     
        class_names=cfg.CLASS_NAMES,    #to be predicted class names defined in .yaml of model
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=None, training=False    
    ) # dist=dist_test #logger=logger

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    
    model.load_params_from_file(filename=model_file_path, logger=logger)
    model.cuda()  # Move model to GPU

    # set model in mode for inference
    model.eval()

    det_annos = [] #list to store the detected annotations
    prediction_dicts = [] #list to store the prediction dictionaries


    # Start inference to retrieve results 
    #### Get the batch_dict from the dataloader
    #Forward pass requires the batch_dict which can be retrieved from the dataloader which is a output of build_dataloader
    for i, batch_dict in enumerate(inference_dataloader):
        load_data_to_gpu(batch_dict) #converts the data to the torch tensors
        with torch.no_grad():

            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box 
            
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

        # For visualization purposes 
        prediction_dicts.append(pred_dicts)


        # Generate the prediction dictionaries to receive class names and BBox coordinates
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=None
            )

        # Append the annos to the detected annotations list
        det_annos += annos

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


    # VISUALIZATION for the first element 

    visualization_frame_data_dict = inference_dataset[5] #get the first element of the dataset
    visualization_flag = True

    # gt_database_file = '../data/kitti/kitti_dbinfos_train_mm.pkl'
    kitti_infos_val_file = '../data/kitti/kitti_infos_val.pkl'

    with open(kitti_infos_val_file, 'rb') as f:
        kitti_infos_val = pickle.load(f)

    ## Retrieve the location of the center, lwh and rotation of the 3D Bounding Box
    selected_frame = 5

    # Load the Database Infos for train
    # The GT BB are provided in the camera coordinate system, see: https://medium.com/towards-data-science/kitti-coordinate-transformations-125094cd42fb
    gt_boxes_in_camera_cf = kitti_infos_val[selected_frame]['annos']['gt_boxes_lidar']

    ##### Reconstructing gt_boxes by hand #####

    # Define the size of the array: 
    num_valid_boxes = sum(1 for name in kitti_infos_val[selected_frame]['annos']['name'] if name != 'DontCare')
    gt_boxes_loc = np.full((num_valid_boxes, 3), -1000.0, dtype=np.float32)
    gt_boxes_dim = np.full((num_valid_boxes, 3), -1.0, dtype=np.float32)
    gt_boxes_rots = np.full(num_valid_boxes, -1000.0, dtype=np.float32)
    valid_index = 0

    # Dont consider gt_boxes if kitti_infos_val[selected_frame]['annos']['name'] == DontCare
    for i in range(len(kitti_infos_val[selected_frame]['annos']['name'])): # iterate over all BB in the frame
        
        # add the content of gt_boxes_loc = kitti_infos_val[selected_frame]['annos']['location'] if the name is not DontCare to a array 
        if kitti_infos_val[selected_frame]['annos']['name'][i] != 'DontCare':
            gt_boxes_loc[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['location'][i])
            gt_boxes_dim[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['dimensions'][i])
            gt_boxes_rots[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['rotation_y'][i])
            valid_index += 1

    R_mats_y = get_rotation_matrix_from_y(gt_boxes_rots)

    # Create a 4x3 matrix for the pose of the 3D BB (rotation + location) 
    gt_boxes_pose_cam = np.zeros((R_mats_y.shape[0], 4, 4)) #R_mats_y.shape[0] to get the number of BB in frame as the first dimension 
    gt_boxes_pose_cam[:, :3, :3] = R_mats_y                 #
    gt_boxes_pose_cam[:, :3, 3] = gt_boxes_loc              # store location in fourth column
    gt_boxes_pose_cam[:, 3, 3] = 1                          # set the element in the last row and last column to 1  

    print("Pose of the first 3D BB in camera coordinates: \n", gt_boxes_pose_cam[0])

    Tr_velo_to_cam = kitti_infos_val[selected_frame]['calib']['Tr_velo_to_cam']
    # print("Tr_velo_to_cam: \n", Tr_velo_to_cam) # for all bb in the frame

    # Transform the 3D BB from cam to vel cf by using the inverse of the Tr_velo_to_cam matrix
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    # print("Tr_cam_to_velo: \n", Tr_cam_to_velo)

    # Print if inverse is correct => Should by identity matrix or really close  
    # print("Tr_cam_to_velo * Tr_velo_to_cam: \n", np.dot(Tr_cam_to_velo, Tr_velo_to_cam))

    gt_boxes_pose_velo = gt_boxes_pose_cam @ Tr_cam_to_velo.T 
    # print("Pose of the first 3D BB in velodyne coordinates: \n", gt_box_pose_velo[0])

    # Extract the location, dimensions and rotation around y-axis from the gt_box_pose_velo
    gt_boxes_in_velodyne_cf = np.zeros((gt_boxes_pose_velo.shape[0], 7)) # 7 for location, dimensions and rotation around y-axis
    gt_boxes_in_velodyne_cf[:, :3] = gt_boxes_pose_velo[:, :3, -1]       # location from last three rows, last column to the first three columns of gt_box

    print("gt_boxes_in_velodyne_cf: \n", gt_boxes_in_velodyne_cf[0])

    gt_boxes_in_velodyne_cf[:, 3:6] = gt_boxes_dim
    # retrieve the rotation_y in velodyne coordinate frame

    gt_boxes_rots_from_velodyne = extract_rotation_y_from_matrices(gt_boxes_pose_velo)

    gt_boxes_in_velodyne_cf[:, -1] = gt_boxes_rots_from_velodyne

    # Convert the gt_boxes_in_velodyne_cf to a torch tensor
    # gt_boxes_in_velodyne_cf = torch.tensor(gt_boxes_in_velodyne_cf).cuda()

    print("GT_Box in velodyne coordinate frame for first box: \n", gt_boxes_in_velodyne_cf[0])

    print("current development")




    # gt_boxes_lidar= box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, Tr_velo_to_cam)

    # calib_file_dir = '../data/kitti/training/calib/'

    # calib_file_path = kitti_common.get_calib_path(kitti_infos_val[selected_frame]['image']['image_path'])
    # calib_file_path = calib_file_dir + kitti_infos_val[selected_frame]['image']['image_idx'] + '.txt'
    # gt_boxes_in_velodyne_cf = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_in_camera_cf, calib_file_path)

    if visualization_flag is not False:
        print("----------------- VISUALIZATION -----------------")    
        print("Visualized frame: ", det_annos[selected_frame]['frame_id'])
        print("Visualized prediction bounding boxes: ", det_annos[5]['name'])
        
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
        print("gt_boxes_in_velodyne_cf: \n", gt_boxes_in_velodyne_cf)
        compare_arrays(gt_boxes_in_velodyne_cf, gt_boxes_in_camera_cf)

        print("----- VIZ in native coordinate system ------")    
        # VisOpen3D.draw_scenes(
        #     points=visualization_frame_data_dict['points'][:, :3],      #input points as part of frame_dict
        #     gt_boxes=gt_boxes_in_camera_cf,
        #     ref_boxes=det_annos[selected_frame]['boxes_lidar'],            # # ref_boxes=pred_dicts[2]['pred_boxes'] -> könnte man auch aus den pred_dicts holen, wenn man die Berechnung der boxes_lidar verfolgt
        #     ref_labels=None,                                            #det_annos[selected_frame]['name'],     # optional: pred_dicts[0]['pred_labels'],   
        #     ref_scores=None                                         # optional: pred_dicts[0]['pred_scores']     # optional
        #     )
        
        print("----- VIZ in Velodyne coordinate system system ------")    
        VisOpen3D.draw_scenes(
            points=visualization_frame_data_dict['points'][:, :3],      
            gt_boxes=gt_boxes_in_velodyne_cf,
            ref_boxes=det_annos[selected_frame]['boxes_lidar'],            
            ref_labels=None,                                            
            ref_scores=None                                         
            )
        
                

    print("Inference finished")

if __name__ == '__main__':
    main()