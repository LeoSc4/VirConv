import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from tools.visual_utils import vis_utils_ls
from tools.visualization.iw_vis import visualize_scene, load_kitti_labels_in_velo
from tools.visual_utils.vis_utils_ls import load_kitti_calib

import datetime
import warnings
warnings.filterwarnings("ignore")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="cfgs/models/kitti/VirConv-T.yaml", help='specify the config for inference')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for inference')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    # parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)
    return args, cfg

def format_annos_for_vis(annos): #transform annos in velo cf for visulization 
    # annos can contain multiple frames (see xx_dataset.py -> generate_prediction_dicts)
    formatted_boxes = []
    for anno in annos: 

        # Get the calib for the selected frame 
        calib_path_for_selected_frame = f"/workspace/data/kitti/testing/calib/{str(anno['frame_id']).zfill(6)}.txt"
        calib_for_selected_frame = load_kitti_calib(calib_path_for_selected_frame)

        formatted_boxes_per_frame = []
        for bbox_idx in range(len(anno['name'])):
            # Pre-formatting 
            center = anno['location'][bbox_idx, :]
            center = np.append(np.array(anno['location'][bbox_idx], dtype=np.float32), 1.0) #add 1.0 for homogenous coordinates
            center_velo = calib_for_selected_frame['cam_rect_to_velo'] @ center
            center_velo = [center_velo[0], center_velo[1], center_velo[2]/2] # + 0.05]   ## ADPT
            rotation_y_velo = np.pi - anno['rotation_y'][bbox_idx]  # Convert from camera frame to lidar frame 
            size = anno['dimensions'][bbox_idx, :]  # original format: l, w, h -> see boxes3d_lidar_to_kitti_camera in kitti_dataset_mm.py
            # change from l, w, h to w, l, h
            size = [size[1], size[0], size[2]]  # width, length, height ->  see boxes3d_lidar_to_kitti_camera in kitti_dataset_mm.py

            formatted_boxes_per_frame.append({
                'type': anno['name'][bbox_idx],
                'dimensions': size,
                'location': center_velo,
                'rotation_y': rotation_y_velo,
                'score': anno['score'][bbox_idx]
                })

        formatted_boxes.append({
            'frame_id': anno['frame_id'],
            'bboxes': formatted_boxes_per_frame
        })

    return formatted_boxes 

def get_pred_boxes_for_frame(det_annos_velo, selected_frame):

    for frame_data in det_annos_velo:
        if frame_data['frame_id'] == selected_frame:
            return frame_data['bboxes']  # Return the bounding boxes for the selected frame
    return []  # Return an empty list if the frame_id is not found

def get_points_for_frame(selected_frame, point_cloud_range=None):
    # selected frame e.g. '000000'
    # point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]

    points_path = f"/workspace/data/kitti/testing/velodyne/{str(selected_frame).zfill(6)}.bin"
    points= np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)   #load from bin in testing

    if point_cloud_range is not None:
        mask = (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) \
            & (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4]) \
            & (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] <= point_cloud_range[5])
    points = points[mask] 

    return points




def main(log_file, model_ckpt, point_cloud_range=None, bbox_analysis_path=None):

    args, cfg = parse_config()
    
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)  #Logger is required for .load_params_from_file function

    # Build the dataloader for inference
    inference_dataset, inference_dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,    #dataset config defined in .yaml of model -> dataset     
        class_names=cfg.CLASS_NAMES,    #to be predicted class names defined in .yaml of model
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=None, training=False    
    )

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=model_ckpt, logger=logger)
    model.cuda()  
    model.eval() # set model in mode for inference

    det_annos = [] #list to store the detected annotations

    print("------------ Starting Inference to retrieve results -------------")
    #Forward pass requires the batch_dict. It can be retrieved from the dataloader which is a output of build_dataloader
    for i, batch_dict in enumerate(inference_dataloader):
        load_data_to_gpu(batch_dict) #converts the data to the torch tensors
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box 
        
        # Generate the prediction dictionaries to receive class names and BBox coordinates
        print("INFO - Generating 'annos' as prediction dictionaries in Camera coordinates")  # annos can contain dicts for multiple frames
            # generate_prediction_dicts applies WBF to the predictions
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=bbox_analysis_path
            ) 

        print("INFO - Preparing det_annos_velo in Velodyne Coordinates from 'annos' for visualization.") # see load_kitti_labels_in_velo in iw_vis.py        
        det_annos_velo = format_annos_for_vis(annos)
        
        # Optional: Enable command line prints
        cli_prints =  False
        if cli_prints == True:
            vis_utils_ls.cl_prints(batch_dict, pred_dicts, det_annos, i)

    print("------------ Starting Visualization -------------")
    for i in range(len(inference_dataset)):
        selected_frame = inference_dataset[i]['frame_id']
        print("INFO - Visualizing the scene for frame %s" % selected_frame)
                
        points = get_points_for_frame(selected_frame, point_cloud_range=point_cloud_range)
        
        # Extract the pred_boxes for the selected frame
        pred_boxes_for_selected_frame = get_pred_boxes_for_frame(det_annos_velo, selected_frame)

        visualize_gt = True 
        gt_labels = None
        if visualize_gt:
            labels_path = f"/workspace/data/kitti/testing/label_2/{str(selected_frame).zfill(6)}.txt"
            calib_path = f"/workspace/data/kitti/testing/calib/{str(selected_frame).zfill(6)}.txt"
            gt_labels = load_kitti_labels_in_velo(labels_path, calib_path)

        visualize_scene(points, gt_labels=gt_labels, predicted_bboxes=pred_boxes_for_selected_frame) #gt_labels=inference_dataset
    


if __name__ == '__main__':

    # Set current working directory 
    os.chdir('/workspace/tools') 

    # Mock command-line arguments 
    sys.argv = [
    'iw_inference_and_vis.py',
    # '--cfg_file', '/workspace/tools/cfgs/models/kitti/VirConv-T-IW-DS-7.yaml',        #for iw_custom_data
    '--cfg_file', '/workspace/tools/cfgs/models/kitti/VirConv-T.yaml',                  #for kitti_reference_data
    '--batch_size', '1',
    '--workers', '0'
    ]
    args = parse_config()
    print(args)

    log_dir = 'workspace/inference' 
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('log_inference_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    model_ckpt = '../output/pretrained_models/VirConv-T-Paper.pth'     

    # Change the point cloud range to visualize only a specific are: [x_min, y_min, z_min, x_max, y_max, z_max]
        # KITTI = [0, -40, -3, 70.4, 40, 1] 
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]          


    # define output path for each frame to write the predictions in a file 
    # integrate the checkpoint name in the path to distinguish the results and also the current time
    bbox_analysis_path = Path(f'/workspace/tools/zz_log_3dbbox_analysis/VirConv-T-Paper.pth/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    bbox_analysis_path.mkdir(parents=True, exist_ok=True) 


    main(log_file, model_ckpt, point_cloud_range, bbox_analysis_path)