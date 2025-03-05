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

# own visualization functions
from tools.visual_utils import vis_utils_ls

import pickle

import datetime

import warnings
warnings.filterwarnings("ignore")

# This inference visualization equals to the meeting notes from 26.02. 
## WIP ##  - It aims to equal to the kitti_vis_standlone.py for the VirConv pipeline integration

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
    return args, cfg


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
    
    model.eval() # set model in mode for inference

    det_annos = [] #list to store the detected annotations
    prediction_dicts = [] #list to store the prediction dictionaries


    # Start inference to retrieve results 
    #Forward pass requires the batch_dict. It can be retrieved from the dataloader which is a output of build_dataloader
    for i, batch_dict in enumerate(inference_dataloader):
        load_data_to_gpu(batch_dict) #converts the data to the torch tensors
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box 
        
        # Optional: Enable command line prints
        cli_prints =  False
        if cli_prints == True:
            vis_utils_ls.cl_prints(batch_dict, pred_dicts, det_annos, i)

        # For visualization purposes 
        prediction_dicts.append(pred_dicts)

        # Generate the prediction dictionaries to receive class names and BBox coordinates
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=None
            )
        # Append the annos to the detected annotations list
        det_annos += annos


#########################   VISUALIZATION    ########################################

    # VISUALIZATION for a selected frame (e.g. frame 5)
    visualization_flag = True
    selected_frame = 5
    visualization_frame_data_dict = inference_dataset[selected_frame] #get the first element of the dataset

    kitti_infos_val_file = '../data/kitti/kitti_infos_val.pkl'
    with open(kitti_infos_val_file, 'rb') as f:
        kitti_infos_val = pickle.load(f) 

    # Load the Database Infos for train
    # The GT BB are provided in the camera coordinate system, see: https://medium.com/towards-data-science/kitti-coordinate-transformations-125094cd42fb
    gt_boxes_in_camera_cf = kitti_infos_val[selected_frame]['annos']['gt_boxes_lidar']

    ##### Reconstructing gt_boxes by hand #####
    # Define the size of the array according to the number of valid bboxes
    num_valid_boxes = sum(1 for name in kitti_infos_val[selected_frame]['annos']['name'] if name != 'DontCare')
    gt_boxes_loc = np.full((num_valid_boxes, 3), -1000.0, dtype=np.float32)
    gt_boxes_dim = np.full((num_valid_boxes, 3), -1.0, dtype=np.float32)
    gt_boxes_rots = np.full(num_valid_boxes, -1000.0, dtype=np.float32)
    
    # Dont consider gt_boxes if kitti_infos_val[selected_frame]['annos']['name'] == DontCare
    valid_index = 0
    for i in range(len(kitti_infos_val[selected_frame]['annos']['name'])): # iterate over all BB in the frame        
        # add the relevant gt_boxes to the arrays 
        if kitti_infos_val[selected_frame]['annos']['name'][i] != 'DontCare':
            gt_boxes_loc[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['location'][i])
            
            #Modification to match with CAH&LS-Vis-Standalone 
            # location y requires - 0.90
            gt_boxes_loc[valid_index][1] -= 0.9 #y is index 1             ##### MANUAL Modification
            
            gt_boxes_dim[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['dimensions'][i])

            #Modification to match with CAH&LS-Vis-Standalone
            # Rotation_y requires + 90 degrees
            gt_boxes_rots[valid_index] = np.array(kitti_infos_val[selected_frame]['annos']['rotation_y'][i])
            # gt_boxes_rots[valid_index] += np.pi / 2 
            valid_index += 1

    R_mats_y = vis_utils_ls.get_rotation_matrices_from_y(gt_boxes_rots)

    # Create a homogeneous 4x4 matrix for the pose of the 3D BB (rotation + location) 
    gt_boxes_pose_cam = np.zeros((R_mats_y.shape[0], 4, 4)) # R_mats_y.shape[0] to get the number of BB in frame as the first dimension 
    gt_boxes_pose_cam[:, :3, :3] = R_mats_y                 # store rotation in first three rows and columns
    gt_boxes_pose_cam[:, :3, 3] = gt_boxes_loc              # store location in fourth column for first three rows (x,y,z)
    gt_boxes_pose_cam[:, 3, 3] = 1                          # set the element in the last row and last column to 1  

    print("Pose of the first 3D BB in camera coordinates: \n", gt_boxes_pose_cam[0])

    Tr_velo_to_cam = kitti_infos_val[selected_frame]['calib']['Tr_velo_to_cam']
    # print("Tr_velo_to_cam: \n", Tr_velo_to_cam) # for all bb in the frame
    R0_rect = kitti_infos_val[selected_frame]['calib']['R0_rect'] # get the rectification matrix for the camera

    Tr_velo_to_cam_rect = R0_rect @ Tr_velo_to_cam
    
    # Get Inverse to transform from cam to velo as visualization is in velo cf
    # Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    # print("Tr_cam_to_velo: \n", Tr_cam_to_velo)
    Tr_cam_rect_to_velo = np.linalg.inv(Tr_velo_to_cam_rect)

    # gt_boxes_pose_velo = Tr_cam_to_velo @ gt_boxes_pose_cam 
    gt_boxes_pose_velo = Tr_cam_rect_to_velo @ gt_boxes_pose_cam 
    
    # print("Pose of the first 3D BB in velodyne coordinates: \n", gt_box_pose_velo[0])

    # Extract the location, dimensions and rotation around y-axis from the gt_box_pose_velo
    print("------Get the GT Boxes for velo cf in correct format------")
    gt_boxes_in_velo_cf = np.zeros((gt_boxes_pose_velo.shape[0], 7)) # 7 for location, dimensions and rotation around y-axis
    gt_boxes_location = gt_boxes_pose_velo[:, :3, -1]       # location from first three rows, last column to the first three columns of gt_box_velo_cf
    gt_boxes_in_velo_cf[:, :3] = gt_boxes_location
    
    # Set box extents/ dimensions (need to adjust for Open3D convention)    
    # KITTI: height, width, length
    # Open3D expects: width, height, length

    gt_boxes_dim_open3d = gt_boxes_dim[:, [1, 0, 2]]    # swap the dimensions to fit the Open3D convention: height to second position, width to first position, length remains
    gt_boxes_in_velo_cf[:, 3:6] = gt_boxes_dim_open3d                     # dimensions remain the same (transform does not change something)

    #### OUR Visualization script adds the 90 degrees here to the rot_y ABOOOOOOVE  ####
    gt_boxes_rots = vis_utils_ls.extract_rotation_y_from_matrices(gt_boxes_pose_velo)
    gt_boxes_in_velo_cf[:, -1] = gt_boxes_rots
    print("GT_Box in velodyne coordinate frame for first box: \n", gt_boxes_in_velo_cf[0])

    if visualization_flag is not False:
        vis_utils_ls.visualize_pc_bbox_results(selected_frame, det_annos, gt_boxes_in_camera_cf, gt_boxes_in_velo_cf, visualization_frame_data_dict)
        print("Inference finished")

if __name__ == '__main__':
    main()