import numpy as np
import matplotlib.pyplot as plt


def calc_log_odds(probability):
    return np.clip(np.log(probability / (1 - probability)), 0, 7) #clipped to [0,7] to avoid extreme values


class CameraCone:
    def __init__(self, hfov, z_min, z_max, resolution):
        self.hfov = hfov
        self.z_min = z_min
        self.z_max = z_max  #maximum visibility in meters 
        self.resolution = resolution  # Converts Meters to Pixels (Ratio)

    def define_bounds(self):
        metric_width = 2 * self.z_max * np.tan(self.hfov / 2)
        template = np.zeros(
            (int(self.resolution * self.z_max), int(self.resolution * metric_width))
        )
        return template

    def gaussian_2d(self, mean, covariance_matrix, point):
        return np.exp(
            -0.5
            * np.dot(
                np.dot((point - mean).T, np.linalg.inv(covariance_matrix)),
                (point - mean),
            )
        )  # / (2 * np.pi * np.sqrt(np.linalg.det(covariance_matrix)))

    def find_visible_points(self):
        # Intialize Template and Points List
        template = self.define_bounds()     #based on the camera parameters
        visible_points = []

        # Define Working Variables
        x_origin = int(template.shape[1] / 2)
        y_origin = template.shape[0]
        max_distance = self.z_max * self.resolution
        min_distance = self.z_min * self.resolution
        fov_slope = y_origin / x_origin                 # Angle between the legs of the FOV 

        # Iterate through each pixel in the template
        for y_pixel in range(template.shape[0]):
            for x_pixel in range(template.shape[1]):
                # Centering wrt to origin  (camera origin instead of map origin)
                x = x_pixel - x_origin
                y = y_origin - y_pixel
                distance = np.sqrt(x**2 + y**2) #of picel to the camera origin
                if (
                    y >= abs(x * fov_slope)         # Check if pixel inside the FoV (left and right corner of the camera)
                    and distance <= max_distance    
                    and distance >= min_distance
                ):
                    # Mark template with 1 for visualization and add to visible points
                    template[y_pixel, x_pixel] = 1
                    visible_points.append((x, y))
        return template, visible_points

    def find_prob_distribution(self): 
        # provde a probability to each points in FoV rating how well it can be seen 
        # Points in Center are higher rated as closer to fov boundaries
        
        normal_dist_mean = [0, 0]
        normal_dist_cov = [[0.5, 0], [0, 0.5]] #cariance matrix with isotropic distribution and variance of 0.5 (-> shape of a bell)
        template = self.define_bounds()
        points_with_prob = []

        _, visible_points = self.find_visible_points()
        for point in visible_points:
            x, y = point
            
            # calculate radius and angle to as cf info
            radius = np.sqrt(x**2 + y**2) / (
                self.z_max * self.resolution
            )  # TO make covariance logical
            angle = np.arctan2(y, x) - (np.pi / 2)

            # Calculate probability of observation quality per point (radius and angle form value in 2D Gaussian space)
            # as closer the point is in the middle, the higher the probability
            probability = self.gaussian_2d(
                normal_dist_mean, normal_dist_cov, np.array([radius, angle])
            )
            log_odd = calc_log_odds(probability) # use to logs to add more easily and boost stability
            points_with_prob.append((x, y, probability, log_odd))

            # For Visualization
            template[
                template.shape[0] - int(y), int(x) + int(template.shape[1] / 2)
            ] = log_odd

        return template, points_with_prob


if __name__ == "__main__":
    # Camera Parameters in Meters & Radians
    horizonal_fov = 65 * np.pi / 180
    z_min = 28 / 100
    z_max =  6 #3
    resolution = 1 / 0.05

    # Create Camera Object
    camera = CameraCone(horizonal_fov, z_min, z_max, resolution)

    # Find Visible Points
    visualizer, visible_points = camera.find_visible_points()
    print("Bounds Shape: ", visualizer.shape)
    print("Visible Points: ", len(visible_points))
    print("Percentage Coverage: ", 100 * (len(visible_points) / (visualizer.size)))

    # Visualize Results
    plt.imshow(visualizer, origin="upper")
    plt.show()
    print(visible_points)

    # Visualize Probability Distribution
    prob_distribution, points_with_prob = camera.find_prob_distribution()
    plt.imshow(prob_distribution, origin="upper")
    plt.show()
    print(points_with_prob)