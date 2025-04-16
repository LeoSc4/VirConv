import json
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from map_coverage_calculator.MapCoverageCalculator import MapCoverageCalculator
from map_coverage_calculator.build_vision_cone import CameraCone

def load_localization_region(json_path):
    with open(json_path, "r") as f:
        region = json.load(f)
        return region["top_left"], region["bottom_right"], region["area"]

def load_camera_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        return data["camera_positions"], data["camera_rotations"]

def main():
    # Set paths
    path_to_map = "../map_coverage_calculator/occupancy_grid.png"
    region_path = "../map_coverage_calculator/localization_region.json"
    poses_path = "../map_coverage_calculator/camera_poses.json"
    output_path = "../map_coverage_calculator/output"

    os.makedirs(output_path, exist_ok=True)

    # Load map and region info
    map_image = cv2.imread(path_to_map)
    top_left, bottom_right, area = load_localization_region(region_path)

    # Camera model
    focal_length = 18.5  # mm
    horizontal_aperture = 36.0  # mm
    vertical_aperture = 10.42  # mm
    h = 1.45  # height of camera in m

    horizonal_fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))        # original =65 * np.pi / 180
    
    vertical_fov_rad = 2 * np.arctan(vertical_aperture / (2 * focal_length))

    theta = vertical_fov_rad / 2    #angle between axis of camera and the lower edge of the field of view

    # z_min = distance in front of sensor from which the lower edge of the field of view touches the ground 
    # assuming camera is parallel to the ground
    z_min = h * np.tan(theta)   #original: 28 

    print(f"z_min ≈ {z_min:.3f} m")

    z_max = 16  #based on point_cloud_range     #3.0

    resolution = 1 / 0.05
    
    camera = CameraCone(horizonal_fov, z_min, z_max, resolution)
    _, visible_points = camera.find_prob_distribution()

    # Init calculator 
    calc = MapCoverageCalculator(
        resolution=resolution,
        visible_points=visible_points,
        image=map_image,
        region_info=[top_left, bottom_right, area],
    )

    # Load poses and apply
    camera_positions, camera_rotations = load_camera_poses(poses_path)

    for pos, rot in zip(camera_positions, camera_rotations):
        transform = {
            "translation": pos,
            "rotation": list(R.from_euler("zyx", rot, degrees=True).as_quat())
        }
        calc.update_pose(transform)

    # Save final coverage
    if calc.overlayed_map is not None:
        cv2.imwrite(os.path.join(output_path, "coverage.png"), calc.overlayed_map)
    if calc.overlayed_map_prob is not None:
        cv2.imwrite(os.path.join(output_path, "probabilistic_coverage.png"), calc.overlayed_map_prob)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
