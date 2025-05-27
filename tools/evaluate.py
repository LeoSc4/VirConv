import argparse
import datetime
import torch
import sys
import os

import numpy as np
from pathlib import Path
from pcdet.utils import common_utils

from pcdet.config import cfg, log_config_to_file, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

from tools.visualization.iw_vis import load_kitti_labels_in_cam

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils


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


def get_bbox_format_from_gt(box): 
    # required format for iou3d_nms_utils.boxes_iou3d_gpu: [x, y, z, dx, dy, dz, heading]
    # KITTI dimensions: h, w, l
    w, l, h = box['dimensions']
    x, y, z = box['location']
    rotation_y = box['rotation_y']
    return [x, y, z, l, w, h, rotation_y]

def get_bbox_format_from_pred(box): 
    # required format for iou3d_nms_utils.boxes_iou3d_gpu: [x, y, z, dx, dy, dz, heading]
    # KITTI dimensions: h, w, l
    l, h, w = box['dimensions']
    x, y, z = box['location']
    rotation_y = box['rotation_y']
    return [x, y, z, l, w, h, rotation_y]


def evaluate_model(logger, gt_labels_cam_for_all_frames, annos_for_all_frames, cfg, log_dir): 
    print("---------- Starting EVALUATION -------------")
    # Evaluate the classification results with a Precision Recall curve 
    # PR in 3DOD is a combination of classification quality and localization-based validation (via IoU)
        # TP is only given if class is correct and spatial overlap is above a certain threshold (IoU)
    
    # Prepare gt_dicts to access in order to align frame idx for GT and prediction
    gt_dict = {}     
    pred_dict = {}    

    # Prepare GT
    for gt_frame in gt_labels_cam_for_all_frames:
        if len(gt_frame) == 0:
            continue
        frame_id = gt_frame[0]['frame_id']  # Annahme: alle in der Liste gehören zum selben Frame
        gt_dict[frame_id] = gt_frame

    # Prepare preds
    for pred_frame in annos_for_all_frames:
        if len(pred_frame) == 0:
            continue
        frame_id = pred_frame[0]['frame_id']  # prediction dict
        pred_dict[frame_id] = pred_frame

    # Set the tresholds 
    iou_threshold = 0.9
    score_threshold = 0.0

    y_true_all = []
    y_score_all = []

    # Iterate over union of GT and prediction frame IDs 
    common_frame_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))

    for frame_id in common_frame_ids:
        gt_frame = gt_dict[frame_id]
        pred_frame = pred_dict[frame_id]

        # print current used frame 
        print(f"Processing frame: {frame_id}") # frame_id is the same for GT and prediction dicts    
        # Prepare the GT
        gt_boxes = [get_bbox_format_from_gt(gt) for gt in gt_frame if gt['type'] == 'Car'] #get the gt_boxes if the type is 'Car' == 'Trolley'
        gt_boxes = torch.tensor(gt_boxes).float().cuda() if gt_boxes else torch.empty((0, 7)).cuda()
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool).cuda()
        
        preds = pred_frame[0]
        pred_boxes = []
        pred_scores = []

        # Filter predictions 
        for i in range(len(preds['name'])):
            print("Current idx of predicted bbox: ", i)
            if preds['name'][i] != 'Car':
                continue
            score = preds['score'][i]
            print("Current score of predicted bbox: ", score)

            if score < score_threshold:
                continue
            pred_box = {
                'location': preds['location'][i],
                'dimensions': preds['dimensions'][i],
                'rotation_y': preds['rotation_y'][i]
            }
            pred_box_tensor = torch.tensor(get_bbox_format_from_pred(pred_box)).unsqueeze(0).float().cuda()
            pred_boxes.append(pred_box_tensor)
            pred_scores.append(score)
        
        # Match the predictions and gt  
        for pred_box, score in zip(pred_boxes, pred_scores):
            if gt_boxes.shape[0] == 0:
                y_true_all.append(0)  # Kein GT vorhanden, also FP
                y_score_all.append(score)
                continue

            print("Current evaluated pred_box idx: ", pred_box)
            # Calculate the IoU for the pred_box compared to all gt_boxes
            ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_box, gt_boxes).squeeze(0)  # (N_gt,)
            
            # Select the maximum IoU and the corresponding index
            max_iou, gt_idx = torch.max(ious, dim=0)
            print("Current max IoU: ", max_iou)

            # Append the pred_box as TP if the IoU is above the threshold and the gt_box is not already matched
            if max_iou >= iou_threshold and not matched_gt[gt_idx]:
                y_true_all.append(1)  # True Positive
                matched_gt[gt_idx] = True
            else:
                y_true_all.append(0)  # False Positive

            y_score_all.append(score)

    # Evaluation over all frames 
    # sklearn.metrics.precision_recall_curve detects FN implicit as it's konwn how many positive labels (y_true_all ==1) are there
    #  if FN should be analysed explicitly: sum them at end of frame loop or sum them over all for FN-Statistic
    
    # Iterate over all possible thresholds existing in y_score_all && calculuate Precision + Recall
    # it uses a sweep over all possible score-values from the predictions
    precision, recall, thresholds = precision_recall_curve(y_true_all, y_score_all)
    
    # Average precision = area under the precision-recall curve
    ap_score = average_precision_score(y_true_all, y_score_all)

    plt.plot(recall, precision, label=f'Trolley (3D IoU AP={ap_score:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("3D Precision-Recall Curve for 'Trolley_RU2'")
    plt.legend()
    plt.grid()

    PR_Curve_path = os.path.join(log_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_PR_curve.png')
    
    # get RECALL_TRESH_LIST from cfg 
    recall_thresholds = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
    precisions_at_recalls = []
    # Iterate over the recall thresholds
    for r_thresh in recall_thresholds:
        # valid indices = all tresholds that are greater than the recall threshold
        valid_indices = np.where(recall >= r_thresh)[0]
        
        if len(valid_indices) == 0:
            precision_at_r = 0.0
        else:
            precision_at_r = max(precision[i] for i in valid_indices)

        precisions_at_recalls.append(precision_at_r)
        logger.info(f"Precision @ Recall ≥ {r_thresh:.2f}: {precision_at_r:.3f}")

    # Mean Average Precision over Recall-Schwellen (mAP@recall)
    mAP_at_recalls = np.mean(precisions_at_recalls)
    logger.info(f"mAP (mean AP over recalls {recall_thresholds}): {mAP_at_recalls:.3f}")

    logger.info(f"Logging the: {precisions_at_recalls}")
    i = 0
    for r_thresh, p_at_r in zip(recall_thresholds, precisions_at_recalls):
        # Find the index of the closest recall value to the threshold
        idx = np.argmin(np.abs(recall - r_thresh))
        plt.scatter(recall[idx], precision[idx], color='red')
        offset = (0, 10) if i % 2 == 0 else (0, -15)

        plt.annotate(
            f'P={precision[idx]:.2f}\nR={recall[idx]:.2f}',   
            (recall[idx], precision[idx]),
            textcoords="offset points",
            xytext=offset,
            ha='center',
            fontsize=8,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
        )
        plt.savefig(PR_Curve_path, dpi=300)
        logger.info(f"PR-curve saved at path: {PR_Curve_path}" )
        i += 1

    print("Amount of entries in y_true_all:", len(y_true_all))
    print("Amount of entries in y_score_all:", len(y_score_all))
    print("Positive labels (y_true_all == 1):", sum(y_true_all))

    logger.info("----------Evaluation finished--------")


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
    gt_labels_cam_for_all_frames = [] 


    #Forward pass requires the batch_dict. It can be retrieved from the dataloader which is a output of build_dataloader
    for i, batch_dict in enumerate(inference_dataloader):


        print("----------- Loading GT labels for batch -------------")
        current_frame = batch_dict['frame_id']
        print("Current frame id of batch: ", current_frame)
        
        for frame_id in current_frame:
            # frame_id_str = str(frame_id).zfill(6)

            if os.path.exists(f"../data/kitti/training/label_2/{frame_id}.txt"):
                labels_path = f"../data/kitti/training/label_2/{frame_id}.txt"
                gt_labels_cam = load_kitti_labels_in_cam(labels_path)
                gt_labels_cam_for_all_frames.append(gt_labels_cam)
            else:
                logger.error(f"ERROR - GT labels in cam cf not found for frame {frame_id}. Skipping evaluation of this ground truth frame.")

        gt_labels_cam_for_all_frames.append(gt_labels_cam) 


        print("----------- Starting Inference for batch-------------")        

        load_data_to_gpu(batch_dict) #converts the data to the torch tensors
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)    #forward pass
                                                                    # batch_dict can be neglected for Bounding Box

        # print("----------- Starting GENERATE PREDICTION DICTS -------------")
        
        # Generate the prediction dictionaries to receive class names and BBox coordinates
        # print("INFO - Generating 'annos' as prediction dictionaries in Camera coordinates")  # annos can contain dicts for multiple frames
            # generate_prediction_dicts applies WBF to the predictions
        annos = inference_dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=bbox_analysis_path
            ) 

        annos_for_all_frames.append(annos) #append the annos for all frames to the list
  
    evaluate_model(logger, gt_labels_cam_for_all_frames, annos_for_all_frames, cfg, log_dir) #evaluate the model with the gt_labels and predictions
    
    


if __name__ == '__main__': 
    
    sys.argv = [
    'evaluate.py',
    '--cfg_file', 'cfgs/models/kitti/VirConv-T-IW-DS-9.yaml',                                               #for iw_custom_data
    # '--cfg_file', '/home/user/workspace/tools/cfgs/models/kitti/VirConv-T-Debug.yaml',                    #for kitti_reference_data
    '--batch_size', '1',
    '--workers', '0'
    ]
    args = parse_config()
    print(args)

    ##### Change if you use kitti/ iw_data #####
    log_dir = 'eval_logs/iw_data9' 

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('%s_log_evaluation.txt' % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    ### BEST RESULT ###     SWP_DS8_2492j5kv: https://wandb.ai/idealworks-ml/VirConv/runs/5gobr6s1?nw=nwuseredgeai
    # IW_DS8_50EP_SWP_DS8_2492j5kv_2492j5kv
    # Epoch 30 = 0.63977
    # model_ckpt = '../output/models/kitti/VirConv-T-IW-DS-8/IW_DS8_50EP_SWP_DS8_2492j5kv_2492j5kv/ckpt/checkpoint_epoch_30.pth'

    ###### Dataset 9 - WITHOUT Augmentor ###########
    ### BEST RESULT ###     SWP_DS9_No-AUG_krx6i4fc: https://wandb.ai/idealworks-ml/VirConv/runs/r35t9lty/overview
    # Epoch 51 = 1.945
    model_ckpt = '../output/models/kitti/VirConv-T-IW-DS-9/IW_DS9_60EP_AUG_SWP_DS9_AUG_krx6i4fc/ckpt/checkpoint_epoch_51.pth'

    ### Change the point cloud range to visualize only a specific are: [x_min, y_min, z_min, x_max, y_max, z_max]
            # KITTI = [0, -40, -3, 70.4, 40, 1]
            # IW_Custom = [0, -16, -3, 16, 16, 1] 
    point_cloud_range = None #[0, -16, -3, 16, 16, 1]   


    main(log_file, model_ckpt, point_cloud_range)