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

# import os
# import time


def main():


    # prev_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    # while True:
    #     current_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    #     print("Current CUDA_VISIBLE_DEVICES: ", current_value)
    #     if current_value != prev_value:
    #         print(f"CUDA_VISIBLE_DEVICES hat sich geändert: {prev_value} -> {current_value}")
    #         prev_value = current_value
    #     time.sleep(1)  # Alle 1 Sekunde prüfen

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

    # Start inference to retrieve results 

    det_annos = [] #list to store the detected annotations



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

    # Create the predicted bounding boxes from det_annos

    ## Retrieve the location of the center, lwh and rotation of the 3D Bounding Box
    # center = det_annos[2]['location']

    selected_frame = 5

    
    # Load the Database Infos for train


    if visualization_flag is not False:
        print("----------------- VISUALIZATION -----------------")    

        print("Visualized frame: ", det_annos[selected_frame]['frame_id'])
        print("Visualized prediction bounding boxes: ", det_annos[5]['name'])
        # output the picture of the visualization_frame_data_dict 
        
        VisOpen3D.draw_scenes(
            points=visualization_frame_data_dict['points'][:, :3],      #input points as part of frame_dict
            gt_boxes=None,
            ref_boxes=det_annos[selected_frame]['boxes_lidar'],            # # ref_boxes=pred_dicts[2]['pred_boxes'] -> könnte man auch aus den pred_dicts holen, wenn man die Berechnung der boxes_lidar verfolgt
            ref_labels=None, #det_annos[selected_frame]['name'],     # optional: pred_dicts[0]['pred_labels'],   
            ref_scores=None     # optional: pred_dicts[0]['pred_scores']     # optional
            )

    print("Inference finished")

if __name__ == '__main__':
    main()