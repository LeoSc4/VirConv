import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from tools.eval_utils import eval_utils
from pcdet.utils import common_utils

from tools.visual_utils import open3d_vis_utils as VisOpen3D
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
    # parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)
    return args, cfg


def main():

    print("Current working directory: ", Path.cwd())
    
    args, cfg = parse_config()
 
    # Create logging
    log_dir = 'workspace/inference' 
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('log_inference_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    #Logger is required for .load_params_from_file function
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    model_file_path = '../output/pretrained_models/VirConv-T-Paper.pth'     #'/workspace/output/models/kitti/VirConv-T-IW-Dataset-7/IW_Dataset_7_EP60_Only_000000/ckpt/checkpoint_epoch_60.pth'     #'../output/pretrained_models/VirConv-T-Paper.pth'

    # Build the dataloader for inference
    inference_dataset, inference_dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,    #dataset config defined in .yaml of model -> dataset     
        class_names=cfg.CLASS_NAMES,    #to be predicted class names defined in .yaml of model
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=None, training=False    
    ) # dist=dist_test #logger=logger

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    
    model.load_params_from_file(filename=model_file_path, logger=logger)
    model.cuda()  

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
        

        # For visualization purposes 
        prediction_dicts.append(pred_dicts)


        # define output path for each frame to write the predictions in a file 
        # integrate the checkpoint name in the path to distinguish the results and also the current time

        bbox_analysis_path = Path(f'/workspace/tools/zz_log_3dbbox_analysis/VirConv-T-Paper.pth/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

        # Create path directories if path is not existent yet 
        bbox_analysis_path.mkdir(parents=True, exist_ok=True)

        # Generate the prediction dictionaries to receive class names and BBox coordinates
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=bbox_analysis_path
            )
        
        # Append the annos to the detected annotations list
        det_annos += annos

        # Optional: Enable command line prints
        cli_prints =  True
        if cli_prints == True:
            vis_utils_ls.cl_prints(batch_dict, pred_dicts, det_annos, i)


#########################   VISUALIZATION    ########################################
"""
    # VISUALIZATION for a selected frame (e.g. frame 5)
    visualization_flag = False #True
    selected_frame = 0
    visualization_frame_data_dict = inference_dataset[selected_frame] #get the first element of the dataset

    kitti_infos_val_file = '../data/kitti/kitti_infos_val.pkl'

    gt_boxes_in_camera_cf, gt_boxes_in_velo_cf = vis_utils_ls.get_gt_boxes_in_velo_cf(kitti_infos_val_file, selected_frame)


    ####### WIP 19.03.2025 -> This requires an update to be displayed correctly ########
    if visualization_flag is not False:
        vis_utils_ls.visualize_pc_bbox_results(selected_frame, det_annos, gt_boxes_in_camera_cf, gt_boxes_in_velo_cf, visualization_frame_data_dict)
        print("Inference finished")
"""
if __name__ == '__main__':
    main()