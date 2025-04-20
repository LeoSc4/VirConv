import os
import copy
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from .build_vision_cone import CameraCone


def inverse_log_odds(log_odds):
    return 1 - 1 / (1 + np.exp(log_odds))


def rotation_to_yaw(rotation):
    return np.rad2deg(R.from_matrix(rotation).as_euler("zyx")[0])


class MapCoverageCalculator:
    def __init__(self, resolution, visible_points, image, region_info):
        # Map Properties
        self.resolution = resolution
        self.map_rgb = image
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        # self.total_visible_in_roi = 0  # visible points (incl. obstacles) 
        self.total_mapped_in_roi = 0   # only visible points (excl. obstacles)
        self.visible_in_roi_mask = np.zeros((self.image_height, self.image_width), dtype=bool)



        # Camera Properties
        self.visible_points = visible_points

        # Localization Region
        self.top_left = tuple(region_info[0])
        self.bottom_right = tuple(region_info[1])
        # self.area = region_info[2]
        self.area = (self.bottom_right[0] - self.top_left[0]) * (self.bottom_right[1] - self.top_left[1])                   ############## CHANGED ACHTUNG!!!

        self.mapped_region = np.zeros((self.image_height, self.image_width))
        self.mapped_log_odds = np.zeros((self.image_height, self.image_width))
        self.mapped_pixels = 0

        # Robot Pose
        self.current_robot_pose = self.previous_robot_pose = None

        # Transformation between map to image frame
        self.image_T_map_matrix = self.transform_to_matrix({
            "translation": [self.image_width / self.resolution, self.image_height / self.resolution, 0.0],
            "rotation": [0.0, 0.0, -1.0, 0.0]  # 180 deg rotation around z-axis
        })

        # Draw the localization region
        cv2.rectangle(self.map_rgb, self.top_left, self.bottom_right, (0, 255, 0), 2, 8)

        # For Saving the Images Whilst Shutdown
        self.overlayed_map = None
        self.overlayed_map_prob = None

    def transform_to_matrix(self, transform):

        # Print the current input transform 
        print("Transform:", transform)


        translation_matrix = np.array(transform["translation"]) * self.resolution
        rotation_matrix = R.from_quat(transform["rotation"]).as_matrix()
        homogeneous_matrix = np.identity(4)
        homogeneous_matrix[:3, :3] = rotation_matrix[:3, :3]
        homogeneous_matrix[:3, 3] = translation_matrix
        return homogeneous_matrix

    def check_in_bounds(self, point):
        x, y = int(point[0]), int(point[1])
        return (
            0 <= x < self.image_width and
            0 <= y < self.image_height
        )
    def check_in_localization(self, point):
        return (
            self.top_left[0] <= point[0] <= self.bottom_right[0] and
            self.top_left[1] <= point[1] <= self.bottom_right[1]
        )

    def check_obstacle(self, point, border_color=(0, 255, 0)):
        x, y = int(point[0]), int(point[1])
        color = self.map_rgb[y, x]
        return np.all(color >= 200)
        # obstacle_colors= {np.all(color >= 200), border_color}     #{(255, 255, 255), border_color}
        # return color in obstacle_colors  # bright areas are rated as free                               


    def check_motion(self):
        if self.current_robot_pose is None or self.previous_robot_pose is None:
            return True
        current_translation = self.current_robot_pose[:3, 3]
        previous_translation = self.previous_robot_pose[:3, 3]
        current_rotation = rotation_to_yaw(self.current_robot_pose[:3, :3])
        previous_rotation = rotation_to_yaw(self.previous_robot_pose[:3, :3])
        delta_translation = np.linalg.norm(current_translation - previous_translation)
        delta_rotation = np.abs(current_rotation - previous_rotation)
        return delta_translation > 0.0 or delta_rotation > 0.0

    def probablistic_coverage(self, threshold):
        required_region = self.mapped_probabilities[
            self.top_left[1]:self.bottom_right[1],
            self.top_left[0]:self.bottom_right[0],
        ]
        return 100 * np.sum(required_region > threshold) / required_region.size

    def calculate_entropy(self):
        required_region = self.mapped_probabilities[
            self.top_left[1]:self.bottom_right[1],
            self.top_left[0]:self.bottom_right[0],
        ]
        return -np.sum(required_region * np.log2(required_region)) / required_region.size

    def overlay_image(self, overlay, image, text, alpha=0.5):
        image = copy.deepcopy(image)
        overlay_normalized = (overlay - 0.5) / 0.5
        overlay_colormap = cv2.applyColorMap((overlay_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_colormap = cv2.cvtColor(overlay_colormap, cv2.COLOR_BGR2RGB)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        blended_image = cv2.addWeighted(image, 1 - alpha, overlay_colormap, alpha, 0)
        image[
            self.top_left[1]:self.bottom_right[1],
            self.top_left[0]:self.bottom_right[0],
        ] = blended_image[
            self.top_left[1]:self.bottom_right[1],
            self.top_left[0]:self.bottom_right[0],
        ]
        image = cv2.putText(
            image,
            text,
            (self.top_left[0] - 20, self.top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        print("Mapped pixels:", np.sum(self.mapped_region))

        return image

    def update_pose(self, map_T_camera_transform):
        image_T_camera = self.image_T_map_matrix @ self.transform_to_matrix(map_T_camera_transform)
        current_pose_viewer = self.map_rgb.copy()
        self.current_robot_pose = self.transform_to_matrix(map_T_camera_transform)
        robot_moved = self.check_motion()

        #Debug 
        pxls_mapped_before = self.mapped_pixels

        for visible_point in self.visible_points:
            point = np.array([visible_point[0], 0, visible_point[1], 1])
            image_T_point = image_T_camera @ point
            ix, iy = int(image_T_point[0]), int(image_T_point[1])

            # Check pixel is valid
            if not (self.check_in_bounds(image_T_point) and self.check_in_localization((ix, iy))):
                continue

            # Count theoretical visibility once per pixel
            if not self.visible_in_roi_mask[iy, ix]:
                self.visible_in_roi_mask[iy, ix] = True

            if self.check_obstacle(image_T_point):  # pixel is free space
                if not self.mapped_region[iy, ix]:
                    self.mapped_region[iy, ix] = 1
                    self.mapped_pixels += 1
                    self.total_mapped_in_roi += 1

                if robot_moved:
                    log_odd = visible_point[3]
                    self.mapped_log_odds[iy, ix] += log_odd

                # Optional: draw
                current_pose_viewer = cv2.circle(
                    current_pose_viewer,
                    (ix, iy),
                    1,
                    (125, 125, 255),
                    1,
                )
        
        print(f"Pose done → newly mapped pixels: {self.mapped_pixels - pxls_mapped_before}")
        print(f"Total mapped pixels: {self.mapped_pixels}")


        self.mapped_log_odds = np.clip(self.mapped_log_odds, 0, 7)
        self.mapped_probabilities = inverse_log_odds(self.mapped_log_odds)
        self.previous_robot_pose = self.current_robot_pose

        self.overlayed_map = self.overlay_image(
            self.mapped_region,
            self.map_rgb,
            f"Coverage: {100 * self.mapped_pixels / self.area:.2f}",
        )

        self.overlayed_map_prob = self.overlay_image(
            self.mapped_probabilities,
            self.map_rgb,
            f"Coverage: {self.probablistic_coverage(0.7):.2f}, Entropy: {self.calculate_entropy():.2f}",
        )

        print(f"\nFinal Coverage Report:")
        print(f"- ROI area (px):                               {self.area}")
        print(f"- Unique visible points in ROI (incl. obstacles): {np.sum(self.visible_in_roi_mask)}")
        print(f"- Total free and visible points in ROI:        {self.total_mapped_in_roi}")
        print(f"- Absolute coverage (free only):               {100 * self.total_mapped_in_roi / self.area:.2f}%")
        print(f"- Theoretical visibility (incl. obstacles):    {100 * np.sum(self.visible_in_roi_mask) / self.area:.2f}%")


        cv2.imshow("Mapped Region", self.overlayed_map)
        cv2.imshow("Mapped Region Probabilities", self.overlayed_map_prob)
        cv2.waitKey(1)
