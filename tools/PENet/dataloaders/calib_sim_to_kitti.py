from tools.PENet.dataloaders.calibration_kitti import get_calib_from_file

import numpy as np

def transform_P2_sim_to_kitti(sim_calib_path, out_kitti_path):
    # Isaac Sim 3D camera intrinsics
        # Isaac provides the following settings: 
            # focal length = 
            # Aperture: 
                # horizontal aperture  = sensor size in width =
                # vertical aperture = sensor size in height
            # focus distance, fStop, Projection, Stereo Role

    original_calib = get_calib_from_file(sim_calib_path)
    print(original_calib['P2'])


    # Given Omniverse information
    focal_length = 50.0             # in mm
    aperture_horizontal = 20.955    # in mm (Sensorbreite)
    aperture_vertical = 15.2908     # in mm (Sensorhöhe)

    # image resolution
    img_height = 352    # in pixels
    img_width =  1216   # in pixels 

    # 1. Calculate the FoV in angles
    FoV_x = 2 * np.arctan(aperture_horizontal / (2 * focal_length))
    FoV_y = 2 * np.arctan(aperture_vertical / (2 * focal_length))

    # 2. Calculate focal lengths in pixels 
    f_u = img_width / (2 * np.tan(FoV_x / 2)) 
    f_v = img_height / (2 * np.tan(FoV_y / 2))

    # 3. Calculate the center points cu and cv in pixels 
        # Center point was not set at first 
    cu = img_width / 2
    cv = img_height / 2

    # Optional: Set the depth scaling factors 
    depth_scaling = original_calib['P2'][2, 2]   # former value from omniverse =  1*10^-9
        # KITTI has a epipolar stereo-camera-calibration that assumes a linear pixel scaling in the depth 
    depth_scaling = 1.0

    # Optional: Set the depth offset correction factor 
    depth_offset = original_calib['P2'][2, 3]
        # KITTI has a stereo camera with base line. Small offset ensures that close objects are projected correctly and no numerical errors occur
    depth_offset = 0.00498          # former value from omniverse =  0.01

    # Optional: Set the disparity and depth estimation influence factor
    b_x = 0     # denotes the baseline (in meters) between the two cameras
    disp_depth_factor = -f_u * b_x

    # 4. Set the new projection matrix 
    P2_kitti_conventions = np.array([[f_u, 0, cu, disp_depth_factor],
                                     [0, f_v, cv, 0],
                                     [0, 0, depth_scaling, depth_offset]])

    # 5. Update the calibration file in the line 2
    # Reopen the read calibration file and update line 2 with the new P2 matrix
    
    # out_kitti_path = sim_calib_path
    with open(sim_calib_path, 'r') as file:
        lines = file.readlines()
    
    lines[2] = 'P2: ' + ' '.join(map(str, P2_kitti_conventions.flatten())) + '\n'

    # Create a new file for out_kitti_path
    with open(out_kitti_path, 'w') as file: 
        file.writelines(lines)

    print("Tranformed calibration from Isaac Sim conventions to KITTI conventions.")


def main(): 
    sim_calib_path_training = '/workspace/data/kitti/training/calib/000000.txt'
    out_kitti_path = '/workspace/data/kitti/training/calib/000000_sim.txt'
    transform_P2_sim_to_kitti(sim_calib_path_training, out_kitti_path)


if __name__ == "__main__":
    main()

