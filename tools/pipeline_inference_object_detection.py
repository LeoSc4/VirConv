import os
import argparse

import numpy as np
import torch

from pathlib import Path
from math import atan2

from copy import deepcopy
from pcdet.config import cfg, log_config_to_file, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

from pcdet.utils import common_utils
from tools.visualization.iw_vis import visualize_scene, load_kitti_labels_in_velo
from tools.visual_utils.vis_utils_ls import load_kitti_calib

from tools.workspace.pose_reconstruction_omnv import get_USD_cam_pose_in_WORLD, map_class_name, get_asset_path_omnv
from tools.workspace.bbox_sim_post_processing_section_overlaps import section_overlaps_post_processing

import datetime
import warnings
warnings.filterwarnings("ignore")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="cfgs/models/kitti/VirConv-T.yaml", help='specify the config for inference')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for inference')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)
    return args, cfg

def format_annos_for_vis(annos): # Transform annos from KITTI Cam to KITTI Velo
    # annos can contain multiple frames (see xx_dataset.py -> generate_prediction_dicts)
    formatted_boxes = []
    for anno in annos: 

        calib_path_for_selected_frame = f"./data/kitti/training/calib/{str(anno['frame_id']).zfill(6)}.txt"
        calib_for_selected_frame = load_kitti_calib(calib_path_for_selected_frame)

        print(f"DEBUG - Transformation 1 (CAM to VELO): \n{calib_for_selected_frame['cam_rect_to_velo']}")

        formatted_boxes_per_frame = []
        for bbox_idx in range(len(anno['name'])):
            center = anno['location'][bbox_idx, :]
            center = np.append(np.array(anno['location'][bbox_idx], dtype=np.float32), 1.0) 
            center_velo = calib_for_selected_frame['cam_rect_to_velo'] @ center
            center_velo = [center_velo[0], center_velo[1], center_velo[2]]          

            r_y = anno['rotation_y'][bbox_idx]
            rot_z_velo = np.pi - r_y  # Convert from camera frame to lidar frame


            size = anno['dimensions'][bbox_idx, :]  # original format: l, w, h -> see boxes3d_lidar_to_kitti_camera in kitti_dataset_mm.py

            size = [size[2], size[0], size[1]]             #+# Tbi

            formatted_boxes_per_frame.append({
                'type': anno['name'][bbox_idx],
                'dimensions': size,
                'location': center_velo,
                'rotation_z': rot_z_velo,
                'score': anno['score'][bbox_idx],
                'bbox': anno['bbox'][bbox_idx],
                'truncated': anno['truncated'][bbox_idx],
                'occluded': anno['occluded'][bbox_idx],
                'alpha': anno['alpha'][bbox_idx]
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

    inference_dataset, inference_dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,    #dataset config defined in .yaml of model --> dataset     
        class_names=cfg.CLASS_NAMES,    #to be predicted class names defined in .yaml of model
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=None, training=False    
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=model_ckpt, logger=logger)
    model.cuda()  
    model.eval() # set model in inference mode

    annos_for_all_frames = [] 
    det_annos_velo_for_all_frames = []

    print("------------ Starting Inference to retrieve results -------------")
    for i, batch_dict in enumerate(inference_dataloader):
        load_data_to_gpu(batch_dict) #converts data to GPU Torch tensors
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box

        # Generate the prediction dictionaries to receive class names and BBox coordinates
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=bbox_analysis_path
            ) 
    
        # Transform the annotations from KITTI Camera to KITTI Velodyne coordinate frame
        det_annos_velo = format_annos_for_vis(annos)

        print(f"INFO: Detections transformed to KITTI VELODYNE!")
        
        annos_for_all_frames.append(annos) #append the annos (KITTI cam) for all frames to the list
        det_annos_velo_for_all_frames.append(det_annos_velo) #append the det_annos_velo for all frames to the list

    print("------------ Starting Logging of predicted BB in KITTI Velo CF -------------")    
    csv_output_path_BB_velo = f'inference_logs/iw_data9/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Inference_ImageSet_predicted bboxes_kitti_cam_cf.csv'
        
    with open(csv_output_path_BB_velo, 'w') as f:                    
        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_velo, y_velo, z_velo, rotation_z, score, frame_id\n')  

        for anno in det_annos_velo_for_all_frames:
            for bbox in anno[0]['bboxes']:

                print(f"DEBUG: RAW Rotation_z for bbox index {anno[0]['bboxes'].index(bbox)} in degrees: {bbox['rotation_z'] * 180 / np.pi:.2f}")

                f.write('%s, %.1f, %.1f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %s\n' % (
                    bbox['type'],
                    bbox['truncated'],
                    bbox['occluded'],
                    bbox['alpha'],
                    bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3],
                    bbox['dimensions'][0], # h
                    bbox['dimensions'][1], # w
                    bbox['dimensions'][2], # l
                    bbox['location'][0],
                    bbox['location'][1],
                    bbox['location'][2],
                    bbox['rotation_z'],
                    bbox['score'],
                    anno[0]['frame_id']
                ))      

        logger.info(f"Predicted Bounding Boxes in KITTI VELO CF written to {csv_output_path_BB_velo}") 

    # create copy to not overwrite the original VELO annos
    annos_for_all_frames_kitti_velo = deepcopy(det_annos_velo_for_all_frames) 

    Tr_velo_to_USD_cam_Convention = np.array([
                                        [   0,   -1,   0,   0],
                                        [   0,    0,   1,   0],
                                        [  -1,    0,   0,   0],
                                        [   0,    0,   0,   1]
                                    ])
    print(f"DEBUG - Transformation 2 (VELO to USD Cam Convention): \n {Tr_velo_to_USD_cam_Convention}")

    cam_graph_extrinsics_path = f"./SGTD_camera_poses/optimized_cameras.json"                   
    get_USD_CAM_pose_in_WORLD_LIST = get_USD_cam_pose_in_WORLD(USD_cam_in_world_path= cam_graph_extrinsics_path)

    annos_for_all_frames_sim_world = []
    for anno_kitti_velo in annos_for_all_frames_kitti_velo:
        frame_id = anno_kitti_velo[0]['frame_id']
        bboxes = anno_kitti_velo[0]['bboxes']

        # Get frame Transformation from USD Cam to World coordinates for current frame
        Tr_USD_Cam_pose_in_world_curr = None
        for m in get_USD_CAM_pose_in_WORLD_LIST:
            if m['frame_id'] == frame_id:
                Tr_USD_Cam_pose_in_world_curr = m['T_USD_CAM_extrinsics']
                break

        for bbox_idx, bbox in enumerate(bboxes):
            # Get BB center in local frame coordinates (KITTI Velo convention)
            bbox_center = np.append(np.array(bbox['location'], dtype=np.float32), 1.0)
            bbox_rot_z_local_velo = bbox['rotation_z']  # Rotation in local velo coordinates 
            
            bbox_local_pose = np.array([
                    [np.cos(bbox_rot_z_local_velo), -np.sin(bbox_rot_z_local_velo),  0, bbox_center[0]],
                    [np.sin(bbox_rot_z_local_velo),  np.cos(bbox_rot_z_local_velo),  0, bbox_center[1]],
                    [0,                                                          0,  1, bbox_center[2]],
                    [0,                                                          0,  0,              1]
                ], dtype=float)
            
            # Transform bbox from local velo coordinates to USD camera convention
            bbox_USD_convention_local_pose = Tr_velo_to_USD_cam_Convention @ bbox_local_pose  # 4x4 

            print(f"INFO: BBox No. {bbox_idx} transformed to USD Cam Convention: \n{bbox_USD_convention_local_pose}") 

            # Transform bbox from USD convention to World coordinates (include extrinsics)
            bbox_world_pose = Tr_USD_Cam_pose_in_world_curr @ bbox_USD_convention_local_pose  # 4x4
            print(f"DEBUG: BBox World Pose for bbox index {bbox_idx} in frame {frame_id}: \n{bbox_world_pose}")

            bbox['location'] = bbox_world_pose[:3, 3]

            bbox_rot_z = atan2(bbox_world_pose[1][0], bbox_world_pose[0][0])
            bbox['rotation_z'] = bbox_rot_z 

            print(f"DEBUG: Rotation_z for bbox index {bbox_idx} in frame {frame_id} in degrees: {bbox_rot_z * 180 / np.pi:.2f}")

            # Map name and get asset path
            bbox['type'] = map_class_name(bbox['type'])
            bbox['asset_path_omnv'] = str(get_asset_path_omnv(bbox['type']))
        
        annos_for_all_frames_sim_world.append(anno_kitti_velo)

    print("------------ Start Writing predicted Bounding Boxes in World coords BEFORE Post-Processing -------------")        
    csv_output_path_BB_SIM_world = './detections/inference_detections_before_post-processing.csv'
    with open(csv_output_path_BB_SIM_world, 'w') as f:
        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z, score, frame_id, asset_path\n')

        for anno_sim in annos_for_all_frames_sim_world:    
            frame_id = anno_sim[0]['frame_id']
            print("Current frame_id: ", frame_id)
            try:
                for bbox in anno_sim[0]['bboxes']:
                    bbox_2d = bbox['bbox']
                    dims = bbox['dimensions']
                    loc = bbox['location']
                    print("DEBUG - RAW Location: ", loc)

                    # Set z to zero to comply with pre-processed scene 
                    loc[2] = 0.0  

                    f.write('%s, %.1f, %.1f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %s, %s\n' % (
                            bbox['type'],
                            bbox['truncated'],
                            bbox['occluded'],
                            bbox['alpha'],
                            bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
                            dims[1], dims[2], dims[0],  # lhw -> hwl
                            loc[0], loc[1], loc[2],
                            bbox['rotation_z'], 
                            bbox['score'], 
                            frame_id,
                            bbox['asset_path_omnv']
                    ))
            except Exception as e:
                print(f"WARNING: Failed writing frame {frame_id}. Error: {e}")
                continue

        logger.info(f"Predicted Bounding Boxes in World coords written to {csv_output_path_BB_SIM_world}")

    print("------------ Starting POST PROCESSING of predicted BB for section overlaps -------------")        
    csv_output_path_BB_SIM_world_POST_PROCESSED = f'./detections/inference_post-processed_detections.csv'

    sim_bboxes_post_processed = section_overlaps_post_processing(csv_output_path_BB_SIM_world, iou_threshold=0.1) #section overlaps post processing with iou threshold of 0.1
    
    with open(csv_output_path_BB_SIM_world_POST_PROCESSED, 'w') as f:      
        f.write('name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z, score, frame_id, bbox_pp_idx, asset_path\n') #toggle based on usage 
        
        for sim_bbox_ppcd in sim_bboxes_post_processed:       
            # Set z to 0 to comply with pre-processed scene
            sim_bbox_ppcd['z_sim_world'] = 0.0

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
                    sim_bbox_ppcd['rotation_z'], 
                    sim_bbox_ppcd['score'], 
                    sim_bbox_ppcd['frame_id'],
                    sim_bbox_ppcd['bbox_idx'],
                    sim_bbox_ppcd['asset_path']
            ))
        
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
                        logger.info(f"\n ----rotation_z per BBoxes: \n {anno[0]['rotation_z']}")

            # Extract the pred_boxes for the selected frame
            pred_boxes_for_selected_frame = get_pred_boxes_for_frame(det_annos_velo_for_all_frames, selected_frame)

        visualize_gt = True 
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