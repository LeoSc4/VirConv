import json 
import math 
from typing import List, Tuple
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from map_coverage_calculator.MapCoverageCalculator import MapCoverageCalculator
from map_coverage_calculator.build_vision_cone import CameraCone
from sensor_graph_coverage_optimizer import CameraCoverageOptimizer
from copy import deepcopy

# For the Python Debugger -> set environment variables for visualization 
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/local/lib/python3.8/dist-packages/PyQt5/Qt/plugins/platforms"
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
# os.environ["QT_QPA_PLATFORM"] = "xcb"

def load_graph_nodes(json_path: str) -> List[Tuple[float, float, float]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(node['x_pixel'], node['y_pixel']) for node in data['nodes']]

def load_graph_and_roi(json_graph_path: str):
    with open(json_graph_path, 'r') as f:
        graph_data = json.load(f)
    roi_data = graph_data["roi"]
    region_info = [roi_data["top_left"], roi_data["bottom_right"], roi_data["area"]]

    # Convert dicts in tuples
    nodes = [(node["x_pixel"], node["y_pixel"]) for node in graph_data["nodes"]]
    reference_point = graph_data["reference_point"]

    return nodes, region_info, reference_point

def interpolate_graph_nodes(
    nodes: List[Tuple[int, int]],
    map_scale: float,
    max_distance_m: float = 2.0,
    min_distance_m: float = 0.5,
    max_interpolations_per_edge: int = 10
) -> List[Tuple[int, int]]:
    """
    Interpolates additional nodes along edges between given nodes based on distance in meters.

    Args:
        nodes: List of (x_pixel, y_pixel) nodes.
        map_scale: Map scale in meters per pixel (e.g., 0.05 for 5 cm/px).
        max_distance_m: Maximum allowed distance between nodes (in meters).
        min_distance_m: Minimum allowed distance between nodes (in meters).
        max_interpolations_per_edge: Max interpolated points per edge.

    Returns:
        List of (x_pixel, y_pixel) nodes including interpolated nodes.
    """

    interpolated_nodes = []

    # Convert distance thresholds from meters to pixels
    max_distance_px = max_distance_m / map_scale
    min_distance_px = min_distance_m / map_scale

    for i in range(len(nodes) - 1):
        start = np.array(nodes[i])
        end = np.array(nodes[i + 1])
        edge_vector = end - start
        edge_length = np.linalg.norm(edge_vector)

        if edge_length == 0:
            continue  # Avoid division by zero

        # Calculate how many interpolations are needed
        num_interpolations = int(edge_length // max_distance_px)

        if num_interpolations > max_interpolations_per_edge:
            num_interpolations = max_interpolations_per_edge

        interpolated_nodes.append(tuple(start.tolist()))  # Add start node

        if num_interpolations > 0:
            for j in range(1, num_interpolations + 1):
                fraction = j / (num_interpolations + 1)
                interpolated_point = start + fraction * edge_vector
                interpolated_nodes.append(tuple(interpolated_point.tolist()))

    # Add the final node
    interpolated_nodes.append(tuple(nodes[-1]))

    # Filter nodes that are too close to each other (closer than min_distance)
    filtered_nodes = [interpolated_nodes[0]]
    for node in interpolated_nodes[1:]:
        last_node = np.array(filtered_nodes[-1])
        current_node = np.array(node)
        if np.linalg.norm(current_node - last_node) >= min_distance_px:
            filtered_nodes.append(node)

    return filtered_nodes

def compute_tangent_angle(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360 # Normalize to [0, 360]

def generate_orientations_per_node(tangent_angle: float): 
    orientations = [
        tangent_angle,                        
        (tangent_angle + 45) % 360,             
        (tangent_angle - 45) % 360,
        (tangent_angle + 135) % 360,            
        (tangent_angle - 135) % 360,             
        (tangent_angle -180) % 360          
    ]
    return orientations

def generate_candidate_configs(nodes: List[Tuple[int, int]], image_height: int, image_width: int, reference_point, map_scale: float = 0.05) -> List[Tuple[List[float], List[float]]]:
    configs = []

    for i in range(len(nodes)):
        x = (reference_point["x_pixel"] + nodes[i][0]) * map_scale
        y = (reference_point["y_pixel"] + nodes[i][1]) * map_scale
        z = 1.45

        if i < len(nodes) - 1:
            tangent_angle = compute_tangent_angle(nodes[i], nodes[i + 1])
        elif i > 0:
            tangent_angle = compute_tangent_angle(nodes[i - 1], nodes[i])
        else:
            tangent_angle = 0.0

        orientations = generate_orientations_per_node(tangent_angle)

        for o in orientations:
            configs.append(([x, y, z], [0.0, o, 90.0]))

    return configs

def run_greedy_set_cover_with_visualization(
        optimizer: CameraCoverageOptimizer,
        candidate_configs: List[Tuple[List[float], List[float]]],
        region_info: List,
        visible_points,
        resolution: float, 
        image_output_path: str, 
        coverage_threshold: float = 0.95,
        verbose: bool = False
    ) -> List[dict]:

    os.makedirs(image_output_path, exist_ok=True)
    covered = set()
    selected_cameras = []
    camera_candidates = []
    remaining_candidates = []

    # I. Calculate coverage of all candidates 
    for pos, orientation in candidate_configs:
        calc = MapCoverageCalculator(
            resolution=resolution,
            visible_points=visible_points,
            image=optimizer.map_image.copy(),
            region_info=region_info
        )

        # Map position from world meter values to pixel values 
        pos_pixel = [pos[0] * resolution, pos[1] * resolution, pos[2]]

        transform = {
            "translation": pos_pixel,
            "rotation": list(R.from_euler("zyx", orientation, degrees=True).as_quat())
        }
        if verbose:
            print(f"- ROI area (px):                               {calc.area}")
            print(f"- Unique visible points in ROI (incl. obstacles): {np.sum(calc.visible_in_roi_mask)}")
            print(f"- Total free and visible points in ROI:        {calc.total_mapped_in_roi}")

        calc.update_pose(transform, verbose=False)
        covered_pixels = set(zip(*np.where(calc.mapped_region > 0)))

        camera_candidates.append((pos_pixel, orientation, covered_pixels))

    remaining_candidates = deepcopy(camera_candidates) 

    top_left, bottom_right = region_info[0], region_info[1]
    total_area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])     # dynamic calculation as area from region_info[2] can be outdated
    step = 0   

    # II. Greedy Set Cover Algorithm
    camera_selection_order = {}
    while len(covered) / total_area < coverage_threshold and remaining_candidates:
        # Choose the best candidate 
        best_candidate = max(remaining_candidates, key=lambda c: len(c[2] - covered)) # Explanation: c[2] is the set of covered pixels for the candidate
        pos_pixel, orientation, new_covered_pixels = best_candidate
        improvement = new_covered_pixels - covered
        print(f"Step {step}: Best candidate covers {len(improvement)} new pixels.")

        # Get the camera index number of the best candidate from the camera_candidates list 
        camera_idx = camera_candidates.index(best_candidate)
        # append camera index and improvement to the camera_selection_order dictionary
        camera_selection_order[camera_idx] = improvement
        print("--------------------- Camera Selection Updated ---------------------")
        print(f"Step {step}: Camera {camera_idx} selected as best candidate with {len(improvement)} new pixels covered.")
        print(f"Step {step}:")
        print(f"  → Selected camera at {pos_pixel} with orientation {orientation}")
        print(f"  → Coverage after selection: {len(covered | improvement) / total_area:.2%} of localization region")

        print("Camera Selection Order:")
        for idx, covered_pixels in camera_selection_order.items():
            print(f"Camera {idx}: {len(covered_pixels)} new pixels covered")
    
        if not improvement:
            break

        covered |= improvement  # Update the covered set by adding the new pixels via union
        selected_cameras.append({"position": pos, "orientation": orientation})
        remaining_candidates.remove(best_candidate)

        # Save the current step as visualization 
        single_calc = MapCoverageCalculator(
            resolution=resolution,
            visible_points=visible_points,
            image=optimizer.map_image.copy(),
            region_info=region_info
        )
        single_transform = {
            "translation": pos_pixel,
            "rotation": list(R.from_euler("zyx", orientation, degrees=True).as_quat())
        }
        single_calc.update_pose(single_transform, verbose=False)
        cam_x_px = int(pos_pixel[0])
        cam_y_px = int(pos_pixel[1])
        cv2.circle(single_calc.overlayed_map, (cam_x_px, cam_y_px), 6, (255, 0, 255), -1)
        cv2.putText(single_calc.overlayed_map, f"{step}", (cam_x_px + 6, cam_y_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        img_path = os.path.join(image_output_path, f"coverage_step_{step:03}.png")
        if single_calc.overlayed_map is not None:
            cv2.imwrite(img_path, single_calc.overlayed_map)
            print(f"Step {step}: Saved coverage image: {img_path}")

        combined_calc = MapCoverageCalculator(
            resolution=resolution,
            visible_points=visible_points,
            image=optimizer.map_image.copy(),
            region_info=region_info
        )
        for idx, cam in enumerate(selected_cameras): 
            transform = {
                "translation": cam["position"],
                "rotation": list(R.from_euler("zyx", cam["orientation"], degrees=True).as_quat())
            }
            combined_calc.update_pose(transform, verbose=False)
            cam_px = int(cam["position"][0] * resolution), int(cam["position"][1] * resolution)
            cv2.circle(combined_calc.overlayed_map, cam_px, 5, (0, 0, 255), -1)
            cv2.putText(combined_calc.overlayed_map, f"{idx}", (cam_px[0] + 6, cam_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        img_combined_path = os.path.join(image_output_path, f"coverage_combined_{step:03}.png")
        if combined_calc.overlayed_map is not None:
            cv2.imwrite(img_combined_path, combined_calc.overlayed_map)
            print(f"Step {step}: Saved combined coverage image to the path: {img_combined_path}")

        step += 1
        print(" ---------------------- CURRENT COVERAGE REPORT ----------------------")
        print(f"Selected {len(selected_cameras)} cameras for current coverage.")
        print(f"Current coverage: {len(covered) / total_area:.2%} of localization region")

    print("---------------------- Optimization Process Finished ----------------------")
    if len(covered) / total_area >= coverage_threshold:
        print(f"Coverage threshold of {coverage_threshold:.2%} reached.")
    else:
        print("No more candidates available or no improvement possible.")


    print(" ---------------------- FINAL COVERAGE REPORT ----------------------")    
    # Output the final status 
    final_calc = MapCoverageCalculator(
        resolution=resolution,
        visible_points=visible_points,
        image=optimizer.map_image.copy(),
        region_info=region_info
    )
    for idx, cam in enumerate(selected_cameras):
        transform = {
            "translation": cam["position"],
            "rotation": list(R.from_euler("zyx", cam["orientation"], degrees=True).as_quat())
        }
        final_calc.update_pose(transform, verbose=False)
        cam_px = int(cam["position"][0] * resolution), int(cam["position"][1] * resolution)

    final_coverage_percentage = 100 * final_calc.mapped_pixels / total_area
    print(f" Final coverage for selected optimized cameras: {final_coverage_percentage:.2f}% of localization region")

    base_dir = os.path.dirname(os.path.dirname(image_output_path))  # -> ./ (Projektwurzel)
    final_image_output_path = os.path.join(base_dir, 'camera_poses_OPT_output')

    print("------------------------- DEBUG -------------------------")
    print(f"Final image output path: {final_image_output_path}")

    final_img_path = os.path.join(final_image_output_path, "coverage_final_combined_cameras.png")
    if final_calc.overlayed_map is not None:
        cv2.imwrite(final_img_path, final_calc.overlayed_map)
        print(f"Final combined coverage image saved to: {final_img_path}")
    print(" ----------------------------------------------------------------------")

    return selected_cameras

def run_set_cover_with_images(optimizer: CameraCoverageOptimizer, candidate_configs: List[Tuple[List[float], List[float]]], region_info: List, visible_points, resolution: float, output_path: str, coverage_threshold: float = 0.95) -> List[dict]:
    os.makedirs(image_output_path, exist_ok=True)
    covered = set()
    optimized_camera_list = []

    for idx, (pos, ori) in enumerate(candidate_configs):
        calc = MapCoverageCalculator(
            resolution=resolution,
            visible_points=visible_points,
            image=optimizer.map_image.copy(),
            region_info=region_info
        )
        transform = {
            "translation": pos,
            "rotation": list(R.from_euler("zyx", ori, degrees=True).as_quat())
        }
        calc.update_pose(transform, verbose=False)

        newly_covered = set(zip(*np.where(calc.mapped_region > 0))) - covered
        if newly_covered:
            covered |= newly_covered
            optimized_camera_list.append({"position": pos, "orientation": ori})

            img_path = os.path.join(image_output_path, f"coverage_{idx:03}.png")
            if calc.overlayed_map is not None:
                cv2.imwrite(img_path, calc.overlayed_map)
                print(f"Saved coverage image: {img_path}")

        #+# CHANGE - Termination criteria -> to easily reached & no optimization used 

        # if len(covered) / region_info[2] >= coverage_threshold:
        #     break

    return optimized_camera_list

def save_optimized_cameras_to_json(optimized_camera_list: List[dict], image_output_path: str, resolution: float = 0.05, reference_point: dict = None):
    """
    Saves the optimized camera list to a JSON file with 'camera_positions' and 'camera_rotations'.
    """
    camera_positions_pixel = []
    camera_positions = []
    camera_rotations = []

    map_scale = 1.0 / resolution

    # convert camera position from pixel to meter values
    for cam in optimized_camera_list:
        x_abs, y_abs, z_abs = cam["position"]

        # Convert reference point to meter first
        ref_x_m = reference_point["x_pixel"] * map_scale
        ref_y_m = reference_point["y_pixel"] * map_scale

        # Compute relative position in pixels (first convert both to pixels)
        x_rel = (x_abs - ref_x_m) / map_scale
        y_rel = (y_abs - ref_y_m) / map_scale

        z_pixel = z_abs / map_scale
        camera_positions_pixel.append([x_rel, y_rel, z_pixel])
        camera_positions.append([x_abs, y_abs, z_abs])  # world coordinates in meters
        cam_orientations_raw = cam["orientation"]
        camera_rotations.append([90.0, 0.0, cam_orientations_raw[1]])  # consistent with orientation convention

    data = {
        "camera_positions_pixel": camera_positions_pixel,  # relative pixels
        "camera_positions": camera_positions,              # world coordinates in meters
        "camera_rotations": camera_rotations
    }

    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    with open(image_output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Saved optimized cameras to {image_output_path}")

def process_graph(json_path: str, map_scale: float = 0.05 / 1) -> dict:
    
    nodes = load_graph_nodes(json_path)
    camera_dict = {}

    for i in range(len(nodes)): 
        x_pixel, y_pixel = nodes[i] 
        x = x_pixel * map_scale  # Convert pixel coordinates to meters
        y = y_pixel * map_scale  
        z = 1.45        # fixed to sensor height 

        # Determine tangent angle to define orientation
        if i < len(nodes) - 1:
            tangent_angle = compute_tangent_angle(nodes[i], nodes[i + 1])
        elif i > 0:
            tangent_angle = compute_tangent_angle(nodes[i - 1], nodes[i])
        else:
            tangent_angle = 0.0  # Default angle for single-node graph

        orientations = generate_orientations_per_node(tangent_angle)

        # Construct camera dict with camera position and orientation for map_coverage_calculator
        camera_dict[i] = {
            "position": [x, y, z],
            "orientations": [[0.0, o, 90.0] for o in orientations]
        }
    return camera_dict

def run_map_coverage_calculation(camera_dict: dict, path_to_map: str, image_output_path: str):
    # Standalone test function
    """ 
    Computes map coverage for set of camera poses defined in camera dict. 
    """ 

    path_to_map = '../map_coverage_calculator/occupancy_grid.png'
    region_path = '../map_coverage_calculator/roi_for_adtc.json'          #localization_region.json

    map_image = cv2.imread(path_to_map) 
    with open(region_path, 'r') as f:
        region_of_interest = json.load(f)
    top_left, bottom_right, area = region_of_interest['top_left'], region_of_interest['bottom_right'], region_of_interest['area']


    # Define camera model 
    focal_length = 18.5
    horizontal_aperture = 36.0 
    vertical_aperture = 10.42
    height = 1.45

    horizontal_fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))
    vertical_fov = 2 * np.arctan(vertical_aperture / (2 * focal_length))
    z_min = height * np.tan(vertical_fov / 2)
    z_max = 16
    resolution = 1 / 0.05  # e.g. 0.05 m/px → 20 px/m

    # Generate camera/ LiDAR vision points 
    sensor = CameraCone(horizontal_fov, z_min, z_max, resolution)
    _, visible_points = sensor.find_prob_distribution()

    # Initialize coverage calculator
    calc = MapCoverageCalculator(
        resolution=resolution,
        visible_points=visible_points,
        image=map_image,
        region_info=[top_left, bottom_right, area],
    )

    # Apply each camera pose
    for cam in camera_dict.values():
        pos = cam["position"]
        for orientation in cam["orientations"]: 
            transform = {            
                "translation": pos,
                "rotation": list(R.from_euler("zyx", orientation, degrees=True).as_quat())
                }
            calc.update_pose(transform, verbose=False)

    # Save absolute coverage map
    if calc.overlayed_map is not None:
        coverage_path = os.path.join(output_path, "coverage.png")
        cv2.imwrite(coverage_path, calc.overlayed_map)
        print(f"Absolute map coverage image saved to: {coverage_path}")
    else:
        print("No overlayed_map was generated.")

    # Compute and return absolute coverage
    absolute_coverage = 100 * calc.mapped_pixels / calc.area
    print(f"Absolute Coverage: {absolute_coverage:.2f}% of localization region")

    cv2.destroyAllWindows()
    return absolute_coverage


if __name__ == "__main__":

    #set workdir to current workdir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # Load graph and ROI from json file
    json_graph_path = './graph_output/base_graph_from_human_input/graph_and_roi.json' 

    map_image_path =    '../map_coverage_calculator/occumap_warehouse_5cm.png'                #'../map_coverage_calculator/occupancy_grid.png'
    image_output_path = './graph_output/coverage_subsets'
    

    ############  Load necessary data ###############
    map_image = cv2.imread(map_image_path)
    image_height = map_image.shape[0]
    image_width = map_image.shape[1]
    
    nodes, region_info, reference_point = load_graph_and_roi(json_graph_path)

    resolution = 1 / 0.05  # e.g. 0.05 m/px → 20 px/m
    
    ############ Sensor model definition #############
    focal_length = 18.5
    horizontal_aperture = 36.0
    vertical_aperture = 10.42
    height = 1.45
    horizontal_fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))
    vertical_fov = 2 * np.arctan(vertical_aperture / (2 * focal_length))
    z_min = height * np.tan(vertical_fov / 2)
    z_max = 16

    resolution_px_per_m = resolution
    sensor = CameraCone(horizontal_fov, z_min, z_max, resolution_px_per_m)
    _, visible_points = sensor.find_visible_points()
   
    # Interpolate nodes to improve coverage and detections
    nodes_interpolated = interpolate_graph_nodes(
        nodes,
        map_scale=0.05,                 # 5 cm per Pixel
        max_distance_m=2.0,             # max. 2m distance
        min_distance_m=0.5,             # min. 0.5 m distance
        max_interpolations_per_edge=5   # max. 5 additional nodes per edge
    )
    
    candidate_configs = generate_candidate_configs(
        nodes_interpolated, 
        map_scale=0.05, 
        image_height=image_height, 
        image_width=image_width,
        reference_point=reference_point
        )

    optimizer = CameraCoverageOptimizer(
        map_image=map_image,
        region_info=region_info,
        resolution=resolution,
        visible_points=visible_points
        )

    # Use Greedy-Set-Cover with visualization 
    optimized_camera_list = run_greedy_set_cover_with_visualization(
        optimizer=optimizer,
        candidate_configs=candidate_configs,
        region_info=region_info,
        visible_points=visible_points,
        resolution=resolution,
        image_output_path=image_output_path,
        coverage_threshold=0.95,                 # Define coverage threshold!
        verbose=False
    )

    output_path = './camera_poses_OPT_output'
    output_json_path = os.path.join(output_path, "optimized_cameras.json")
    save_optimized_cameras_to_json(optimized_camera_list, output_json_path, resolution, reference_point)


    print("------------------- OUTPUT - OPTIMIZED SENSOR SELECTION -------------------")
    print("\Optimized sensor selection :")
    for idx, cam in enumerate(optimized_camera_list):
        print(f"Camera {idx}: Position={cam['position']}, Orientation={cam['orientation']}")


    # Finished the optimization process
    print("Finished the optimization process.")
    print("---------------------------------------------------------------------------")
