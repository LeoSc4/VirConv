
from pcdet.ops.iou3d_nms import iou3d_nms_utils

import torch


def section_overlaps_post_processing(omnv_world_bbox_poses_path, iou_threshold=0.8):
    """
    Post-process bounding boxes (BB) to handle overlaps between warehouse sections.
    If section overlaps between the different warehouse sections are existent, the BB needs to be post processed.

    The BB are already transformed into a common coordinate system (Omniverse Isaac Sim world coordinates).

    Approach:
    1. Load the BB
    2. Check if the BB overlaps with any other BB
    3. If overlap is above iou_threshold:
    - Check if it is the same class
    4. If overlap > iou_threshold and same class:
    - Remove the BB with the lower classification confidence score

    Args:
        omnv_world_bbox_poses_path (str): Path to the CSV file containing bounding box data.
        iou_threshold (float): IoU threshold above which overlapping BBs are considered.

    Returns:
        list: Filtered bounding boxes after removing overlaps.
    """
    
    # Example structure of the CSV file:
    # name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z_sim, score, frame_id, asset_path
    # Trolley_RU2, 0.0, 0.0, -2.8291, 561.6717, 224.3429, 673.4056, 271.6760, 0.4976, 0.9223, 1.3056, -7.4251, 1.4987, 0.2416, -2.8115, 0.9990, 000000, /home/leo/workspace/Omniverse/OwnAssets/Trolley_RU2/RU2_dolly.usdc

    sim_bboxes = []

    # Load the bounding box poses from the CSV file
    with open(omnv_world_bbox_poses_path, 'r') as f:
        bbox_lines_with_header = f.readlines()
        bbox_lines = [line.strip().split(',') for line in bbox_lines_with_header[1:]]  # Skip the header
        
        # Extract relevant information
        for bbox in bbox_lines:
            name = bbox[0]
            truncated = float(bbox[1])
            occluded = float(bbox[2])
            alpha = float(bbox[3])
            u1, v1, u2, v2 = map(float, bbox[4:8])
            h, w, l = map(float, bbox[8:11])
            x_sim_world, y_sim_world, z_sim_world = map(float, bbox[11:14])
            rotation_z_sim = float(bbox[14])
            score = float(bbox[15])
            frame_id = int(bbox[16])
            asset_path = bbox[17]

            # sim_bboxes: (N, 7) [x, y, z, dx, dy, dz, heading]
            sim_bbox = [x_sim_world, y_sim_world, z_sim_world, l, w, h, rotation_z_sim]

            sim_bboxes.append({
                'bbox_idx': bbox_lines.index(bbox),
                'name': name,
                'truncated': truncated,
                'occluded': occluded,
                'alpha': alpha,
                'u1': u1,
                'v1': v1,
                'u2': u2,
                'v2': v2,
                'h': h,
                'w': w,
                'l': l,
                'x_sim_world': x_sim_world,
                'y_sim_world': y_sim_world,
                'z_sim_world': z_sim_world,
                'rotation_z': rotation_z_sim,
                'score': score,
                'frame_id': frame_id,
                'asset_path': asset_path,
                'sim_bbox': sim_bbox
            })

        # Using iou3d_nms_utils.boxes_iou3d_gpu
        bbox_to_remove = set() # saves all elements only once, no duplicates

        sim_bboxes_tensor = torch.tensor([bbox['sim_bbox'] for bbox in sim_bboxes], dtype=torch.float32).cuda()  # Convert to a tensor and move to GPU

        # sort the bounding boxes by score in descending order
        sorted_indices = torch.argsort(torch.tensor([bbox['score'] for bbox in sim_bboxes]), descending=True)
        sim_bboxes_tensor = sim_bboxes_tensor[sorted_indices]  # Sort the bounding boxes by score
        sim_bboxes = [sim_bboxes[i] for i in sorted_indices.tolist()]  # Sort the bounding boxes by score

        for bbox_idx in range(len(sim_bboxes)):
            for other_bbox_idx in range(bbox_idx + 1, len(sim_bboxes)):
                box_a = sim_bboxes_tensor[bbox_idx].unsqueeze(0)
                box_b = sim_bboxes_tensor[other_bbox_idx].unsqueeze(0)

                iou = iou3d_nms_utils.boxes_iou3d_gpu(box_a, box_b).item()  # Calculate 3D IoU and convert tensor to a scalar value

                # Check if IoU is above the threshold
                if iou > iou_threshold:
                    # Check if they are of the same class (name)
                    if sim_bboxes[bbox_idx]['name'] == sim_bboxes[other_bbox_idx]['name']:
                        # Remove the bounding box with the lower score
                        if sim_bboxes[bbox_idx]['score'] < sim_bboxes[other_bbox_idx]['score']:
                            bbox_to_remove.add(sim_bboxes[bbox_idx]['bbox_idx'])
                        else:
                            bbox_to_remove.add(sim_bboxes[other_bbox_idx]['bbox_idx'])

        # Remove the marked bboxes
        filtered_bboxes = []
        for bbox in sim_bboxes:
            if bbox['bbox_idx'] not in bbox_to_remove:
                filtered_bboxes.append(bbox)

        sim_bboxes_post_processed = filtered_bboxes
        
    return sim_bboxes_post_processed


def main():
    # Example usage
    omnv_world_bbox_poses_path = 'tools/workspace/mocked_omnv_sim_pred_boxes_20250411.csv'
    iou_threshold = 0.1

    section_overlaps_post_processing(omnv_world_bbox_poses_path, iou_threshold) 
    
    
if __name__ == "__main__":
    
    main()
