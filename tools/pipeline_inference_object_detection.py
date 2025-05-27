import os
import argparse

import numpy as np
import torch

from pathlib import Path

from copy import deepcopy
from pcdet.config import cfg, log_config_to_file, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

from pcdet.utils import common_utils
from tools.visualization.iw_vis import visualize_scene, load_kitti_labels_in_velo
from tools.visual_utils.vis_utils_ls import load_kitti_calib

from tools.workspace.pose_reconstruction_omnv import get_camera_pose_omnv_world
from tools.workspace.pose_reconstruction_omnv import reconstruct_bbox_pose_omnv_world
from tools.workspace.pose_reconstruction_omnv import map_class_name
from tools.workspace.pose_reconstruction_omnv import get_asset_path_omnv
from tools.workspace.bbox_sim_post_processing_section_overlaps import section_overlaps_post_processing

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

        calib_path_for_selected_frame = f"./data/kitti/training/calib/{str(anno['frame_id']).zfill(6)}.txt"
        calib_for_selected_frame = load_kitti_calib(calib_path_for_selected_frame)

        formatted_boxes_per_frame = []
        for bbox_idx in range(len(anno['name'])):
            # Pre-formatting 
            center = anno['location'][bbox_idx, :]
            center = np.append(np.array(anno['location'][bbox_idx], dtype=np.float32), 1.0) #add 1.0 for homogenous coordinates
            center_velo = calib_for_selected_frame['cam_rect_to_velo'] @ center
            center_velo = [center_velo[0], center_velo[1], center_velo[2]]          
            rotation_y_velo = np.pi - anno['rotation_y'][bbox_idx]  # Convert from camera frame to lidar frame 
            size = anno['dimensions'][bbox_idx, :]  # original format: l, w, h -> see boxes3d_lidar_to_kitti_camera in kitti_dataset_mm.py

            size = [size[2], size[0], size[1]]  

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
        if frame_data[0]['frame_id'] == selected_frame:
            return frame_data[0]['bboxes']  # Return the bounding boxes for the selected frame
    return []  # Return an empty list if the frame_id is not found

def get_points_for_frame(selected_frame, point_cloud_range=None):
    points_path = f"./data/kitti/training/velodyne/{str(selected_frame).zfill(6)}.bin"
    points= np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)   #load from bin in training

    if point_cloud_range is not None:
        mask = (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) \
            & (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4]) \
            & (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] <= point_cloud_range[5])
        points = points[mask] 

    return points


def main(log_file, model_ckpt, point_cloud_range=None, bbox_analysis_path=None):

    args, cfg = parse_config()
    
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)  #Logger is required for .load_params_from_file function

    logger.info('**********************Start logging**********************')
    log_config_to_file(cfg, logger=logger) #write the complete config to the log file


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

    annos_for_all_frames = [] 
    det_annos_velo_for_all_frames = []

    print("------------ Starting Inference to retrieve results -------------")
    #Forward pass requires the batch_dict. It can be retrieved from the dataloader which is a output of build_dataloader
    for i, batch_dict in enumerate(inference_dataloader):
        load_data_to_gpu(batch_dict) #converts the data to the torch tensors
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box


        # print("----------- Starting GENERATE PREDICTION DICTS -------------")
        
        # Generate the prediction dictionaries to receive class names and BBox coordinates
            # generate_prediction_dicts applies WBF to the predictions
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=bbox_analysis_path
            ) 
        
        det_annos_velo = format_annos_for_vis(annos)
        
        annos_for_all_frames.append(annos) #append the annos for all frames to the list
        det_annos_velo_for_all_frames.append(det_annos_velo) #append the det_annos_velo for all frames to the list

    print("------------ Starting Logging of predicted BB in KITTI Cam CF -------------")    
    csv_output_path_BB_cam = f'inference_logs/iw_data9/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Inference_ImageSet_predicted bboxes_kitti_cam_cf.csv'
    # write annos in one csv with frame_id at last column 
    with open(csv_output_path_BB_cam, 'w') as f:                    
        # format for csv: name, truncated, occluded, alpha, bbox[0], bbox[1], bbox[2], bbox[3], dimensions[0], dimensions[1], dimensions[2], location[0], location[1], location[2], rotation_y, score
        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_cam, y_cam, z_cam, rotation_y, score, frame_id\n') #toggle based on usage 
        
        for anno in annos_for_all_frames:       
            for bbox_idx in range(len(anno[0]['name'])):
                bbox_2d = anno[0]['bbox'][bbox_idx]
                dims = anno[0]['dimensions'][bbox_idx]
                loc = anno[0]['location'][bbox_idx]

                f.write('%s, %.1f, %.1f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %s\n' % (
                        anno[0]['name'][bbox_idx],
                        anno[0]['truncated'][bbox_idx],
                        anno[0]['occluded'][bbox_idx],
                        anno[0]['alpha'][bbox_idx],
                        bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
                        dims[1], dims[2], dims[0], # #lhw -> hwl
                        loc[0], loc[1], loc[2],
                        anno[0]['rotation_y'][bbox_idx], 
                        anno[0]['score'][bbox_idx], 
                        anno[0]['frame_id'] 
                ))
                        
        logger.info(f"Predicted Bounding Boxes in KITTI Cam CF written to {csv_output_path_BB_cam}") 


    print("------------ Starting Logging of predicted BB in Simulation World CF -------------")    

    # Get the predicted BBoxes in camera coordinate frame
    # create copy to not overwrite the original annos
    annos_for_all_frames_kitti_cam = deepcopy(annos_for_all_frames) 
    cam_graph_extrinsics_path = f"./data/kitti/poses_dataset_9.json"       #defined in omniverse isaac sim default camera convention

    # Get the camera extrinsics for all frames 
    Tr_cam_transform_matrices = get_camera_pose_omnv_world(omnv_def_cam_pose_omnv_world_path=cam_graph_extrinsics_path)

    annos_for_all_frames_sim_world = []


    for anno_kitti_cam in annos_for_all_frames_kitti_cam:
        for bbox_idx in range(len(anno_kitti_cam[0]['name'])):
            curr_frame_id = anno_kitti_cam[0]['frame_id']
            # Pre-formatting 
            # center = anno_kitti_cam[0]['location'][bbox_idx, :]
            center_hom = np.append(np.array(anno_kitti_cam[0]['location'][bbox_idx], dtype=np.float32), 1.0)

            bbox_center_omnv_world = reconstruct_bbox_pose_omnv_world(Tr_cam_transform_matrices, pred_bbox_center_kitti_cam=center_hom, current_frame_id=curr_frame_id)

            # update the location of the bbox in the annos
            anno_kitti_cam[0]['location'][bbox_idx, :] = bbox_center_omnv_world[:3]

            # update the rotation_y to rotation_z naming as the bbox are now in Omniverse Isaac Sim world coordinates with z up  
            if bbox_idx == 0: #only delete the array of the rotations (list of rotation_y) once because anno_kitti_cam[0] contains multiple bboxes
                anno_kitti_cam[0]['rotation_z_sim'] = deepcopy(anno_kitti_cam[0]['rotation_y'])
                del anno_kitti_cam[0]['rotation_y']

            # Map the class names to application class names 
            anno_kitti_cam[0]['name'] = anno_kitti_cam[0]['name'].astype('<U20')    
            print(anno_kitti_cam[0]['name'].dtype)
            anno_kitti_cam[0]['name'][bbox_idx] = map_class_name(anno_kitti_cam[0]['name'][bbox_idx])
            anno_kitti_cam[0]['asset_path_omnv'] = str(get_asset_path_omnv(anno_kitti_cam[0]['name'][bbox_idx])) #get the asset path for the specific class name

        annos_for_all_frames_sim_world.append(anno_kitti_cam)

    csv_output_path_BB_SIM_world = f'inference_logs/iw_data9/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Inference_ImageSet_predicted bboxes_SIM_world_cf.csv'
    with open(csv_output_path_BB_SIM_world, 'w') as f:      

        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z_sim, score, frame_id, asset_path\n') #toggle based on usage 
        
        for anno_sim in annos_for_all_frames_sim_world:       
            for bbox_idx in range(len(anno_sim[0]['name'])):
                bbox_2d = anno_sim[0]['bbox'][bbox_idx]
                dims = anno_sim[0]['dimensions'][bbox_idx]
                loc = anno_sim[0]['location'][bbox_idx]

                f.write('%s, %.1f, %.1f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %s, %s\n' % (
                        anno_sim[0]['name'][bbox_idx],
                        anno_sim[0]['truncated'][bbox_idx],
                        anno_sim[0]['occluded'][bbox_idx],
                        anno_sim[0]['alpha'][bbox_idx],
                        bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
                        dims[1], dims[2], dims[0], # #lhw -> hwl
                        loc[0], loc[1], loc[2],
                        anno_sim[0]['rotation_z_sim'][bbox_idx], 
                        anno_sim[0]['score'][bbox_idx], 
                        anno_sim[0]['frame_id'],
                        anno_sim[0]['asset_path_omnv']
                ))
        
        logger.info(f"Predicted Bounding Boxes in SIM World CF written to {csv_output_path_BB_SIM_world}") 

    print("------------ Starting POST PROCESSING of predicted BB for section overlaps -------------")
    
    csv_output_path_BB_SIM_world_POST_PROCESSED = f'inference_logs/iw_data9/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Inference_ImageSet_predicted bboxes_SIM_world_cf_POST_PROCESSED.csv'
    
    sim_bboxes_post_processed = section_overlaps_post_processing(csv_output_path_BB_SIM_world, iou_threshold=0.1) #section overlaps post processing with iou threshold of 0.1
    
    with open(csv_output_path_BB_SIM_world_POST_PROCESSED, 'w') as f:      

        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z_sim, score, frame_id, bbox_pp_idx, asset_path\n') #toggle based on usage 
        
        for sim_bbox_ppcd in sim_bboxes_post_processed:       
            # for bbox_idx in range(len(sim_bboxes_post_processed[0]['name'])):
            f.write('%s, %.1f, %.1f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %06d, %s, %s\n' % (
                    sim_bbox_ppcd['name'],
                    sim_bbox_ppcd['truncated'],
                    sim_bbox_ppcd['occluded'],
                    sim_bbox_ppcd['alpha'],
                    sim_bbox_ppcd['u1'], sim_bbox_ppcd['v1'],
                    sim_bbox_ppcd['u2'], sim_bbox_ppcd['v2'],
                    sim_bbox_ppcd['h'], sim_bbox_ppcd['w'], sim_bbox_ppcd['l'],
                    sim_bbox_ppcd['x_sim_world'], sim_bbox_ppcd['y_sim_world'], sim_bbox_ppcd['z_sim_world'],
                    sim_bbox_ppcd['rotation_z_sim'], 
                    sim_bbox_ppcd['score'], 
                    sim_bbox_ppcd['frame_id'],
                    sim_bbox_ppcd['bbox_idx'],
                    sim_bbox_ppcd['asset_path']
            ))
        
        # print("Amount of post-processed bboxes: ", len(sim_bboxes_post_processed))
        
        logger.info(f"POST PROCESSED Predicted Bounding Boxes in SIM World CF written to {csv_output_path_BB_SIM_world_POST_PROCESSED}") 
  
    visualize_scene_now = False
    if visualize_scene_now: 
        print("------------ Starting Visualization -------------")

        for i in range(len(inference_dataset)):
            selected_frame = inference_dataset[i]['frame_id']
            print("INFO - Visualizing the scene for frame %s" % selected_frame)        
            points = get_points_for_frame(selected_frame, point_cloud_range=point_cloud_range)
                
            logger.info(f"Predicted Bounding Boxes for frame {selected_frame}:")
            logger.info("-> Values are in camera coordinate frame.")                    

            for anno in annos_for_all_frames: 
                    if anno[0]['frame_id'] == selected_frame:
                        logger.info(f"\n ----Dimensions per BBoxes: \n {anno[0]['dimensions']}")
                        logger.info(f"\n ----Locations per BBoxes: \n {anno[0]['location']}")
                        logger.info(f"\n ----Rotation_y per BBoxes: \n {anno[0]['rotation_y']}")

            # Extract the pred_boxes for the selected frame
            pred_boxes_for_selected_frame = get_pred_boxes_for_frame(det_annos_velo_for_all_frames, selected_frame)

        visualize_gt = False #True 
        gt_labels = None
        if visualize_gt:
            # execute only if gt_labels are available, otherwise skip
            if os.path.exists(f"../data/kitti/training/label_2/{str(selected_frame).zfill(6)}.txt"):
                labels_path = f"../data/kitti/training/label_2/{str(selected_frame).zfill(6)}.txt"
                calib_path = f"../data/kitti/training/calib/{str(selected_frame).zfill(6)}.txt"
                gt_labels = load_kitti_labels_in_velo(labels_path, calib_path)
            else:
                logger.warning(f"Ground truth labels not found for frame {selected_frame}. Skipping visualization of ground truth.")

        visualize_scene(points, gt_labels=gt_labels, predicted_bboxes=pred_boxes_for_selected_frame, selected_frame=selected_frame) 
