import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from tools.visual_utils.vis_utils_ls import load_kitti_calib, load_kitti_labels, kitti_to_open3d_bbox

point_cloud_analysis = True
filter_point_cloud_flag = False

bbox_visualization = True

image_analysis = False


if image_analysis:

       image_idx = 2
       image_path = f"/workspace/data/kitti/training/image_2/{str(image_idx).zfill(6)}.png"
       label_path = f"/workspace/data/kitti/training/label_2/{str(image_idx).zfill(6)}.txt"

       img = plt.imread(image_path)
       print("Reading the image:", image_path)
       print("Image Shape:", img.shape)
       
       # Load the labels 
       # label = [type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height_3d, width_3d, length_3d, x_3d, y_3d, z_3d, rotation_y, score]
       labels = load_kitti_labels_raw(label_path)

       # labels is of type [{'bbox_2d': [0.0, ... , ..., ...], 'type': 'Car'}, {'bbox_2d': [0.0, ... , ..., ...]}, {}]
       # bbox_2d_label = labels[0]['bbox_2d']

       # Visualize the image with 2D bounding boxes 
       fig, ax = plt.subplots(1, 1, figsize=(10, 10))
       ax.imshow(img)
       for bbox_label in labels:
              bbox = bbox_label['bbox_2d']
              rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none') # requires xy as tuple, width, height
              ax.add_patch(rect) # add the rectangle to the image
       # plt.axis('off') 
       
       # Create a folder '2D bbox visualized' if not existing and save the fig in there 
       output_dir = '/workspace/tools/zz_testspace/iw_custom_dataset7/2D_bbox_visualized'
       os.makedirs(output_dir, exist_ok=True)
       
       # save the fig with reference to the image path
       output_path = os.path.join(output_dir, os.path.basename(image_path)).replace(".png", "_bbox_2d.png")

       print('Saved the image with 2D BB to path:', output_path)
       plt.savefig(output_path)


if point_cloud_analysis: 
       
       frame_idx = 2
       
       # iw_custom_dataset7 (test)files
       file_path = f"/workspace/data/kitti/training/velodyne/{str(frame_idx).zfill(6)}.bin"
       
       # Path to a KITTI point cloud binary file subsampled training 
       # file_path = "/workspace/tools/zz_testspace/kitti_sample_data/000053.bin"

       # Load the point cloud (KITTI format stores float32 values in x, y, z, intensity order)
       point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
       print("Current evaluated point cloud: ",  file_path)

       """
       # Alternative: Read from .npy file
       # file_path_npy = f"/workspace/tools/zz_testspace/kitti_sample_data/{str(frame_idx).zfill(6)}.bin"             #/workspace/data/kitti/training/velodyne_depth/
       file_path_npy = f"/workspace/data/kitti/training/velodyne_depth/{str(frame_idx).zfill(6)}.npy"             #/workspace/data/kitti/training/velodyne_depth/
       file_path_npy = f"/workspace/data/kitti/training/velodyne_depth/{str(frame_idx).zfill(6)}.npy"             #/workspace/data/kitti/training/velodyne_depth/
       
       point_cloud = np.load(file_path_npy)
       print("Current evaluated NPY point cloud: ",  file_path_npy)
       """

       # Print the shape of the loaded point cloud
       print("Point Cloud Shape:", point_cloud.shape)

       # Point ranges (max - min) in x,y,z 
       print("Point Cloud Range:")
       for i in range(3):
              print(f"{['x', 'y', 'z'][i]}: {point_cloud[:, i].max() - point_cloud[:, i].min()} meters")


       # Create a df from the point cloud and describe 
       pc_df = pd.DataFrame(point_cloud[:,:4], columns=["x", "y", "z", "intensity"])
       print(pc_df.describe())

       # Visualize a histogram for x, y, z, intensity
       pc_df.hist(bins=50, figsize=(20, 15))

       plt.savefig(f'/workspace/tools/zz_testspace/iw_custom_dataset7/pc_histogram_{str(frame_idx).zfill(6)}.png')


       # POINT_CLOUD_RANGE = [x_min, y_min, z_min, x_max, y_max, z_max]
       # x is in driving direction, y is left-right, z is up-down
       # Filter points by range
       # point_cloud_range = [0, -40, -3, 70.4, 40, 1]        #default

       if filter_point_cloud_flag is True: 
              point_cloud_range = [0, -16, -1, 64, 16, 4]         # Can be optimized and decreased !!          

              mask = (point_cloud[:, 0] >= point_cloud_range[0]) & (point_cloud[:, 0] <= point_cloud_range[3]) \
              & (point_cloud[:, 1] >= point_cloud_range[1]) & (point_cloud[:, 1] <= point_cloud_range[4]) \
              & (point_cloud[:, 2] >= point_cloud_range[2]) & (point_cloud[:, 2] <= point_cloud_range[5])
              point_cloud = point_cloud[mask] #apply the mask to the point cloud

              # Print the shape of the filtered point cloud
              print("Filtered Point Cloud Shape:", point_cloud.shape)

              # Point ranges for filtered point cloud(max - min) in x,y,z 
              print("Point Cloud Range in filtered point cloud:")
              for i in range(3):
                     print(f"{['x', 'y', 'z'][i]}: {point_cloud[:, i].max() - point_cloud[:, i].min()} meters")

       # Convert to Open3D point cloud format
       pcd = o3d.geometry.PointCloud()

       pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Use only x, y, z

       intensity = point_cloud[:, 3]
       colors = np.zeros((point_cloud.shape[0], 3))
       colors[:, 0] = intensity  # red channel
       colors[:, 1] = intensity  # green channel
       colors[:, 2] = intensity  # blue channel
       pcd.colors = o3d.utility.Vector3dVector(colors)

       # draw the coordinate system into o3d
       coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

       # Visualize the point cloud with the coordinate frame
       vis = o3d.visualization.Visualizer()
       vis.create_window()
       vis.add_geometry(pcd)
       vis.add_geometry(coordinate_frame)


       if bbox_visualization:     
              calib_path = f"/workspace/data/kitti/training/calib/{str(frame_idx).zfill(6)}.txt"
              calib = load_kitti_calib(calib_path)
              
              label_path = f"/workspace/data/kitti/training/label_2/{str(frame_idx).zfill(6)}.txt"
              labels = load_kitti_labels(label_path)

              for i, bbox_data in enumerate(labels):
                     print(f"\nBox {i+1}: {bbox_data['type']}")
                     print(f"  Camera coordinates: center={bbox_data['location']}, rotation_y={bbox_data['rotation_y']}")
            
                     # Create Open3D bounding box with calibration
                     o3d_bbox = kitti_to_open3d_bbox(bbox_data, calib)
                     vis.add_geometry(o3d_bbox)  

                     location_bbox = bbox_data['location']
                     print("Location of the 3D BB: ", location_bbox)
                     location_bbox = np.append(location_bbox, 1)
                     location_bbox = calib['cam_rect_to_velo'] @ location_bbox
                     location_bbox = location_bbox[:3]
                     location_bbox = np.array([location_bbox[0], location_bbox[1], location_bbox[2]/2])                # divide by 2 to match the center of the bbox
                     location_bbox = location_bbox.reshape(3,1)
                     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                     sphere.translate(location_bbox)
                     sphere.paint_uniform_color([1, 0, 0])

                     vis.add_geometry(sphere)

       # Setze Punktgröße
       opt = vis.get_render_option()
       opt.point_size = 2.0  # Hier kannst du die Punktgröße anpassen

       vis.run()
       vis.destroy_window()
       

""" 
############     IW_custom_dataset7  - test_fix_kitti_correction  ############
       # ##########    000000.bin   ##########
root@leo-Blade-16-RZ09-0510:/workspace# /usr/bin/python /workspace/tools/zz_testspace/data_analysis.py
Reading the image: /workspace/data/kitti/training/image_2/000000.png
Image Shape: (352, 1216, 3)
Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset7/2D_bbox_visualized/000000_bbox_2d.png
Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000000.bin
Point Cloud Shape: (192983, 4)
Point Cloud Range:
x: 43.82807922363281 meters
y: 43.573856353759766 meters
z: 0.4356543719768524 meters
                   x              y              z      intensity
count  192983.000000  192983.000000  192983.000000  192983.000000
mean        4.435950       0.187055      -0.350676       0.557855
std         4.840153       2.920304       0.019789       0.132172
min         1.417965     -21.064404      -0.352000       0.023527
25%         1.823099      -0.993949      -0.352000       0.466620
50%         2.578788       0.092369      -0.352000       0.515976
75%         4.740058       1.186977      -0.352000       0.686206
max        45.246044      22.509453       0.083654       0.872120
Location of the 3D BB:  [0.177268, 0.352, 10.278133]
Velodyne to Camera Rectified:
 [[ 0. -1.  0.  0.]
 [ 0.  0. -1.  0.]
 [ 1.  0.  0.  0.]
 [ 0.  0.  0.  1.]]

Camera Rectified to Velodyne:
 [[ 0.  0.  1.  0.]
 [-1. -0. -0. -0.]
 [-0. -1. -0. -0.]
 [ 0.  0.  0.  1.]]
       

 # ##########    000001.bin   ##########
 root@leo-Blade-16-RZ09-0510:/workspace# /usr/bin/python /workspace/tools/zz_testspace/data_analysis.py
Reading the image: /workspace/data/kitti/training/image_2/000001.png
Image Shape: (352, 1216, 3)
Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset7/2D_bbox_visualized/000001_bbox_2d.png
Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000001.bin
Point Cloud Shape: (207576, 4)
Point Cloud Range:
x: 53.882652282714844 meters
y: 69.72055053710938 meters
z: 0.44143441319465637 meters
                   x              y              z      intensity
count  207576.000000  207576.000000  207576.000000  207576.000000
mean        4.809922      -0.016821      -0.347100       0.441005
std         5.737612       3.748055       0.037498       0.064998
min         1.417962     -38.879982      -0.352001       0.015685
25%         1.864068      -1.163443      -0.352000       0.411724
50%         2.749757       0.014799      -0.352000       0.442646
75%         5.130989       1.144258      -0.352000       0.482305
max        55.300613      30.840570       0.089434       0.686653

Location of the 3D BB:  [-0.697815, 0.352, 5.74031]
Velodyne to Camera Rectified:
 [[ 0. -1.  0.  0.]
 [ 0.  0. -1.  0.]
 [ 1.  0.  0.  0.]
 [ 0.  0.  0.  1.]]

Camera Rectified to Velodyne:
 [[ 0.  0.  1.  0.]
 [-1. -0. -0. -0.]
 [-0. -1. -0. -0.]
 [ 0.  0.  0.  1.]]




############     IW_custom_dataset6  (Teil 2)   ############
       # ##########    000000.bin   ##########
       root@leo-Blade-16-RZ09-0510:/workspace# /usr/bin/python /workspace/tools/zz_testspace/data_analysis.py
       Reading the image: /workspace/data/kitti/training/image_2/000000.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset6/2D_bbox_visualized/000000_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000000.bin
       Point Cloud Shape: (48386, 4)
       Point Cloud Range:
       x: 42.0501823425293 meters
       y: 22.906038284301758 meters
       z: 0.43936434388160706 meters
                     x             y             z  intensity
       count  48386.000000  48386.000000  48386.000000    48386.0
       mean      14.297218     -0.274839     -0.345314        0.0
       std        9.245384      3.075833      0.043424        0.0
       min        4.073609    -12.959995     -0.352000        0.0
       25%        6.907427     -1.741239     -0.352000        0.0
       50%       11.084011     -0.112054     -0.352000        0.0
       75%       19.586811      1.384396     -0.352000        0.0
       max       46.123791      9.946043      0.087364        0.0

############     IW_custom_dataset6  (Teil 1)   ############
       # ##########    000000.bin   ##########
       Reading the image: /workspace/data/kitti/training/image_2/000000.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset6/2D_bbox_visualized/000000_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000000.bin
       Point Cloud Shape: (50230, 4)
       Point Cloud Range:
       x: 33.822635650634766 meters
       y: 42.07529830932617 meters
       z: 0.4402727782726288 meters
                     x             y             z  intensity
       count  50230.000000  50230.000000  50230.000000    50230.0
       mean       0.146978     11.298066     -0.348490        0.0
       std        4.539495      8.582950      0.031937        0.0
       min      -15.840002      2.141555     -0.352000        0.0
       25%       -1.930431      4.395825     -0.352000        0.0
       50%        0.032282      8.445913     -0.352000        0.0
       75%        2.081678     15.993325     -0.352000        0.0
       max       17.982634     44.216854      0.088273        0.0
       Location of the 3D BB:  [0.177268, 0.352, 10.278133]
              




############     IW_custom_dataset6     ############
       # ##########    000000.bin   ##########
root@leo-Blade-16-RZ09-0510:/workspace# /usr/bin/python /workspace/tools/zz_testspace/data_analysis.py
Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000000.bin
Point Cloud Shape: (52713, 4)
Point Cloud Range:
x: 855.35546875 meters
y: 0.4362577497959137 meters
z: 2036.794921875 meters
                  x             y             z  intensity
count  52713.000000  52713.000000  52713.000000    52713.0
mean      -1.893878      0.347843    104.877457        0.0
std       38.510303      0.034137    278.006683        0.0
min     -427.677765     -0.084255      5.819425        0.0
25%       -3.347914      0.352000     13.708898        0.0
50%       -0.225812      0.352000     31.425011        0.0
75%        2.141062      0.352000     81.705009        0.0
max      427.677673      0.352002   2042.614380        0.0
Filtered Point Cloud Shape: (36338, 4)
Point Cloud Range in filtered point cloud:
x: 27.592243194580078 meters
y: 0.4362577497959137 meters
z: 60.0717887878418 meters
Location of the 3D BB:  [1.161381, 0.352, 15.832774]











############     IW_custom_dataset4     ############
       # ##########    000000.bin   ##########

       INFO - 2025-03-08 11:25:24,386 - font_manager - Generating new fontManager, this may take some time...
       Reading the image: /workspace/data/kitti/training/image_2/000000.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset4/2D_bbox_visualized/000000_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000001.bin
       Point Cloud Shape: (194842, 4)
       Point Cloud Range:
       x: 23.149477005004883 meters
       y: 42.0501823425293 meters
       z: 0.4385209381580353 meters
                     x              y              z      intensity
       count  194842.000000  194842.000000  194842.000000  194842.000000
       mean        0.003533       9.997066       0.338587       0.289026
       std         2.252749       7.321055       0.060724       0.078680
       min       -10.930052       4.073609      -0.086521       0.003921
       25%        -1.079082       5.276152       0.352000       0.239192
       50%         0.007681       7.486057       0.352000       0.301206
       75%         1.071440      11.084011       0.352000       0.333300
       max        12.219425      46.123791       0.352000       0.627388

       # ##########    000001.bin   ##########
       Reading the image: /workspace/data/kitti/training/image_2/000001.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset4/2D_bbox_visualized/000001_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000001.bin
       Point Cloud Shape: (194842, 4)
       Point Cloud Range:
       x: 23.149477005004883 meters
       y: 42.0501823425293 meters
       z: 0.4385209381580353 meters
                     x              y              z      intensity
       count  194842.000000  194842.000000  194842.000000  194842.000000
       mean        0.003533       9.997066       0.338587       0.289026
       std         2.252749       7.321055       0.060724       0.078680
       min       -10.930052       4.073609      -0.086521       0.003921
       25%        -1.079082       5.276152       0.352000       0.239192
       50%         0.007681       7.486057       0.352000       0.301206
       75%         1.071440      11.084011       0.352000       0.333300
       max        12.219425      46.123791       0.352000       0.627388
       
       # ##########    000002.bin   ##########
       Reading the image: /workspace/data/kitti/training/image_2/000002.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset4/2D_bbox_visualized/000002_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000002.bin
       Point Cloud Shape: (196407, 4)
       Point Cloud Range:
       x: 29.295486450195312 meters
       y: 53.11991500854492 meters
       z: 0.4407293498516083 meters
                     x              y              z      intensity
       count  196407.000000  196407.000000  196407.000000  196407.000000
       mean        0.065806      10.591962       0.340729       0.517624
       std         2.424717       8.630350       0.055945       0.216089
       min       -13.455483       4.073606      -0.088729       0.000000
       25%        -1.022911       5.276152       0.352000       0.321089
       50%         0.036561       7.486058       0.352000       0.580334
       75%         1.113109      11.438700       0.352000       0.725418
       max        15.840003      57.193520       0.352000       0.866580

       # ##########    000003.bin   ##########
       Reading the image: /workspace/data/kitti/training/image_2/000003.png
       Image Shape: (352, 1216, 3)
       Saved the image with 2D BB to path: /workspace/tools/zz_testspace/iw_custom_dataset4/2D_bbox_visualized/000003_bbox_2d.png
       Current evaluated point cloud:  /workspace/data/kitti/training/velodyne/000003.bin
       Point Cloud Shape: (197553, 4)
       Point Cloud Range:
       x: 28.666120529174805 meters
       y: 58.093265533447266 meters
       z: 0.44048255681991577 meters
                     x              y              z      intensity
       count  197553.000000  197553.000000  197553.000000  197553.000000
       mean       -0.153309      10.804742       0.343325       0.387080
       std         2.404449       9.070665       0.049443       0.073940
       min       -17.107206       4.073608      -0.088482       0.020053
       25%        -1.140126       5.276153       0.352000       0.364222
       50%        -0.041491       7.565274       0.352000       0.392118
       75%         1.047163      12.015439       0.352000       0.435251
       max        11.558914      62.166874       0.352000       0.641454

"""


############         KITTI         ############
       # ##########   Example from KITTI:      000053.bin   ##########

       # Point Cloud Shape: (123279, 4)
       #                    x              y              z      intensity
       # count  123279.000000  123279.000000  123279.000000  123279.000000
       # mean       -0.642432       0.598885      -0.848487       0.266359
       # std        10.425736       5.126933       0.770761       0.136218
       # min       -78.524002     -51.178001      -7.754000       0.000000
       # 25%        -3.932000      -2.455000      -1.557000       0.190000
       # 50%         0.151000       0.496000      -0.937000       0.290000
       # 75%         3.112000       5.880000      -0.202000       0.350000
       # max        79.112999      32.203999       2.878000       0.990000
       # Filtered Point Cloud Shape: (62764, 4)
