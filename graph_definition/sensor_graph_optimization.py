import json 
import math 
from typing import List, Tuple
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from map_coverage_calculator.MapCoverageCalculator import MapCoverageCalculator
from map_coverage_calculator.build_vision_cone import CameraCone


def load_graph_nodes(json_path: str) -> List[Tuple[float, float, float]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(node['x_pixel'], node['y_pixel']) for node in data['nodes']]

def compute_tangent_angle(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360 # Normalize to [0, 360)

def generate_orientations_per_node(tangent_angle: float): 
    orientations = [
        tangent_angle,                          # Tangent
        (tangent_angle + 90) % 360,             # Perpendicular right
        (tangent_angle - 90) % 360,             # Perpendicular left
        (tangent_angle + 135) % 360,            # Diagonal back right
        (tangent_angle - 135) % 360             # Diagonal back left
    ]
    return orientations

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

def run_map_coverage_calculation(camera_dict: dict, path_to_map: str, output_path: str):
    """ 
    Computes map coverage for set of camera poses defined in camera dict. 
    """ 

    path_to_map = '../map_coverage_calculator/occupancy_grid.png'
    region_path = '../map_coverage_calculator/localization_region.json'
    os.makedirs

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
            calc.update_pose(transform)

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

def save_camera_dict_as_json(camera_dict, output_path: str):
    positions = []
    rotations = []
    for entry in camera_dict.values():
        for ori in entry["orientations"]:
            positions.append(entry["position"])
            rotations.append(ori)
    data = {
        "camera_positions": positions,
        "camera_rotations": rotations
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    json_graph_path = './graph_output/graph_iw_warehouse.json'
    
    camera_dict = process_graph(json_graph_path)
    # print the output in a readable format
    for node, camera in camera_dict.items():
        print(f"Node {node}: Position: {camera['position']}, Orientations: {camera['orientations']}")

    map_coverage = run_map_coverage_calculation(camera_dict, path_to_map='../map_coverage_calculator/occupancy_grid.png', output_path='./graph_output')


    
