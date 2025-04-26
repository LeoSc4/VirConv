import numpy as np 
import cv2 
import os 
import json 
from typing import List, Dict, Tuple
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from map_coverage_calculator.MapCoverageCalculator import MapCoverageCalculator
from map_coverage_calculator.build_vision_cone import CameraCone


class CameraCandidate: 
    """
    Saves the position, orientation and covered pixels of a camera candidate (pose)
    """
    def __init__(self, position: List[float], orientation: List[float], covered_pixels: set):
        self.position = position
        self.orientation = orientation
        self.covered_pixels = covered_pixels   



class CameraCoverageOptimizer: 
    """
    simulate_coverage(): uses MapCoverageCalculator to compute coverage of pose 
    greedy_set_cover(): selects from all candidates the ones that cover the most pixels which are not covered yet (step by step)
    run_optimization(): runs the optimization process
    """
    def __init__(self, map_image: np.ndarray, region_info: List[List[int]], resolution: float, visible_points: np.ndarray): 
        self.map_image = map_image
        self.region_info = region_info
        self.resolution = resolution
        self.visible_points = visible_points
        self.area = (region_info[1][0] - region_info[0][0]) * (region_info[1][1] - region_info[0][1])

    def simulate_coverage(self, position: List[float], orientation: List[float]) -> set:

        transform = { 
            "translation": position, 
            "rotation": list(R.from_euler('xyz', orientation, degrees=True).as_quat())     #as Quaternion ??
        }

        calc = MapCoverageCalculator(
            resolution = self.resolution,
            visible_points = self.visible_points,
            image = deepcopy(self.map_image),
            region_info = self.region_info,
        )

        calc.update_pose(transform)

        covered = set(zip(*np.where(calc.mapped_region > 0)))
        return covered

    def greedy_set_cover(self, candidates: List[CameraCandidate], target_coverage_ratio = 0.9) -> List[CameraCandidate]:
        covered = set()
        selected = []
        total_pixels = self.area
        
        while len(covered) / total_pixels < target_coverage_ratio and candidates: 
            best_candidate = max(candidates, key=lambda c:len(c.covered_pixels - covered))
            new_coverage = best_candidate.covered_pixels - covered  

            if not new_coverage:
                break

            selected.append(best_candidate)
            covered |= new_coverage # |= means union of sets 
            candidates.remove(best_candidate)

        return selected
    
    def run_set_cover_optimization(self, candidate_configs: List[Tuple[List[float], List[float]]], target_coverage = 0.9) -> List[Dict]:
        candidates = []

        for pos, orientation in candidate_configs:
            covered_pixels = self.simulate_coverage(pos, orientation)
            candidates.append(CameraCandidate(pos, orientation, covered_pixels))
        
        selected = self.greedy_set_cover(candidates, target_coverage_ratio=target_coverage)

        print(f"Selected {len(selected)} cameras for {target_coverage * 100:.1f}% coverage.")
        return [{"position": cam.position, "orientation": cam.orientation} for cam in selected]
