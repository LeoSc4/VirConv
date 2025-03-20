import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from dataloaders import calibration_kitti
from skimage import io
import cv2

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral

from dataloaders.my_loader import depth2pointsrgb, depth2pointsrgbp

from datetime import datetime

from tools.visual_utils.vis_utils_ls import save_point_cloud_as_pcd

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype('uint8')

def merge_into_row(ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))
        img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        #predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert('RGB'))
        #predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert('RGB'))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert('RGB'))
        img_list.append(extra2)
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename):
    #torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:
        # LS: returns a mask that is True for points that are in the image and in front of the camera
    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)     # bring rectified points to image plane
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1]) # check if x-coordinates of points are in the width of the image 
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0]) # check if y-coordinates of points are in the height of the image
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)   # merge the two flags
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)    # check if points are in front of the camera
    return pts_valid_flag

def save_depth_as_points(depth, idx, root_path): ##########

    
    ########## File Index Preprocessing added to use ImageSets that don't start with 000000 idx or contain  ##########
    #ectract the mode (e.g. 'testing' or 'training' from the path
    type_ImageSet = root_path.split('/')[-1]
    if type_ImageSet == 'testing':
        # open the /workspace/data/kitti/ImageSets/test.txt file and get the index of the image
        # the content is e.g. 
        # with open('/workspace/data/kitti/ImageSets/test.txt', 'r') as f:

        with open('/workspace/data/kitti/ImageSets/test.txt', 'r') as f: # for iw_custom_dataset2
            lines = f.readlines()
            file_idx = int(lines[idx].strip())
    
    elif type_ImageSet == 'training':
        # if the index extends the ImageSet for training, then retrieve the index from the val.txt instead of train.txt
        # with open('/workspace/data/kitti/ImageSets/train.txt', 'r') as f:

        with open('/workspace/data/kitti/ImageSets/train.txt', 'r') as f: # for iw_custom_dataset2
            lines = f.readlines()
            if idx <= len(lines):
                file_idx = int(lines[idx].strip())
            if idx > len(lines):
                idx = idx - len(lines)      # reduce per number of lines to get right index in val.txt

                with open('/workspace/data/kitti/ImageSets/val.txt', 'r') as f:
                    lines = f.readlines()
                    file_idx = int(lines[idx].strip())
    
    file_idx = str(file_idx).zfill(6)
    ###############

    # lightweight option for the training pipeline
    # file_idx = str(idx).zfill(6)


    file_image_path = os.path.join(root_path, 'image_2', file_idx + '.png')
    file_velo_path = os.path.join(root_path, 'velodyne', file_idx + '.bin')
    file_calib = os.path.join(root_path, 'calib', file_idx + '.txt')

    calib = calibration_kitti.Calibration(file_calib)

    lidar = np.fromfile(str(file_velo_path), dtype=np.float32).reshape(-1, 4)
    image = np.array(io.imread(file_image_path), dtype=np.int32)
    
    # https://github.com/JUGGHM/PENet_ICRA2021/issues/10

    image = image[:352, :1216] # crop to 352x1216

    # TEMP 
    ## save the cropped image to check if the cropping is correct

    # integrate the current time into the save path 
    cv2.imwrite(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_cropped_image_{file_idx}.png', image.astype(np.uint8))

    # save the lidar as .pcd file type 
    print("#+# Saving the raw_lidar_from_file as .npy & .pcd file")

    # np.save(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_lidar_from_raw_file.npy', lidar)
    # save_point_cloud_as_pcd(lidar, f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_lidar_from raw_file.pcd')

    pts_rect = calib.lidar_to_rect(lidar[:, 0:3])

    #+# print the pts_rect to check if the points are in the right range (use current date time)
    print(f'BLOCK - Main Iterate in Loop {datetime.now().strftime("%Y%m%d_%H%M%S")}: pts_rect: {pts_rect}')
    print("#+# Saving the pts_rect as .npy & .pcd file")
    # np.save(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_pts_rect_I.npy', pts_rect)
    # save_point_cloud_as_pcd(pts_rect, f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_pts_rect_I.pcd')

    fov_flag = get_fov_flag(pts_rect, image.shape, calib)
    print(f'BLOCK - Main Iterate in Loop {datetime.now().strftime("%Y%m%d_%H%M%S")}: fov_flag: {fov_flag}')   
    print("#+# Saving the get_fov_flag as .npy & .pcd file")
    # np.save(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_fov_flag_II.npy', fov_flag)

    #Commented out the FOV_flag #+#        ### To be tested again
    lidar = lidar[fov_flag]
    print(f'BLOCK - Main Iterate in Loop {datetime.now().strftime("%Y%m%d_%H%M%S")}: Applied the fov_flag to the lidar points')
    #+# save the lidar to check if the points are in the right range (use current date time)
    print("#+# Saving the lidar with applied fov as .npy & .pcd file")
    # np.save(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_lidar_with_applied_fov_III.npy', lidar)
    # save_point_cloud_as_pcd(lidar, f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_lidar_with_applied_fov_III.pcd')
    

    paths = os.path.join(root_path, 'velodyne_depth')
    if not os.path.exists(paths):
        os.makedirs(paths)

    out_path = os.path.join(paths, file_idx + '.npy')
    depth = depth.cpu().detach().numpy().reshape(352, 1216,1)

    # Generating final points before saving as velodyne depth with [N x 8]
    ##
    final_points = depth2pointsrgbp(depth, image, calib, lidar)

    #+# Save the final points to velodyne_depth path
    final_points = final_points.astype(np.float16)
    np.save(out_path, final_points)

    #+# save the final points to the pipeline_investigation check 
    print("#+# Saving the final_points as .npy & .pcd file")
    np.save(f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_final_points_velodyne_depth_IV.npy', final_points)
    # save_point_cloud_as_pcd(final_points, f'/workspace/data/kitti/training/pipeline_investigation/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{file_idx}_final_points_velodyne_depth_IV.pcd')



def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(img, filename):
    #from tensor
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
