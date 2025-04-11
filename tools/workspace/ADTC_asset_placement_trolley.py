from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

import omni.usd
from pxr import Usd, UsdGeom, Gf

import time
import math

import os 

def compute_bbox_center(prim: Usd.Prim) -> Gf.Vec3d:
    """
    Compute the center of the bounding box of a given prim.
 
    Args:
        prim: A Usd.Prim object.
 
    Returns:
        Gf.Vec3d: The center point of the aligned bounding box.
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    min_pt = bound_range.GetMin()
    max_pt = bound_range.GetMax()
    center = (min_pt + max_pt) * 0.5
    return center

def get_Tr_bb_center_to_prim_cf(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Compute the transformation matrix from the bounding box center to the prim's local cf.
    Args:
        prim: A Usd.Prim object.
    
    Returns:
        Gf.Matrix4d: The transformation matrix.
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    min_pt = bound_range.GetMin()
    max_pt = bound_range.GetMax()
    center = (min_pt + max_pt) * 0.5
    translation_mat = Gf.Matrix4d().SetTranslate(center)
    return translation_mat


# Create or get the World
world = World(stage_units_in_meters=1.0)

# Load the stage
stage = omni.usd.get_context().get_stage()

sim_bbox_infos_path = '/home/leo/workspace/Omniverse/2025-04-11_19-21-23_Inference_ImageSet_predicted bboxes_SIM_world_cf.csv'

# For lines in csv except header: 
# Get the asset path 

with open(sim_bbox_infos_path, 'r') as f:
        # The structure is: 'name, truncated, occluded, alpha, u1, v1, u2, v2, h, w, l, x_sim_world, y_sim_world, z_sim_world, rotation_z_sim, score, frame_id, asset_path

    lines = f.readlines()
    idx = 0 
    for bbox_sim in lines[1:]: # Skip the header
        # Split the line by comma
        parts = bbox_sim.strip().split(',')
        # Get the asset path
        bbox_class = str(parts[0])

        asset_path = os.path.abspath(parts[-1].strip()) #ensures that the path is absolute and no whitespace problem
        if not os.path.exists(asset_path):
            print(f"[ERROR] Asset path does not exist: {asset_path}")
            continue
        print("Asset path: " + str(asset_path))
        # Construct the prim path with unique identifier counting upwards for same class
        
        prim_path = f"/World/{bbox_class}_{idx}"
        print("Prim path: " + str(prim_path))
        
        # Add the asset reference to the stage (world) 
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)    
        idx += 1 # increment the index for the next object prim path

        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            print(f"[ERROR] Prim at path {prim_path} is not valid.")
            continue

        xform = UsdGeom.Xformable(prim) # ensure that prim is tranformable

        ## Apply the Translation ##
        # Get the transformation from bbox center to the prim's local coordinate frame
        
        translation_mat = get_Tr_bb_center_to_prim_cf(prim)

        # Combine the translation_mat and the translation from csv file 
        transl_x = float(parts[11]) - translation_mat[3][0] 
        transl_y = float(parts[12]) - translation_mat[3][1] 
        transl_z = float(parts[13]) - translation_mat[3][2] 

        translation = Gf.Vec3d(transl_x, transl_y, transl_z)
        xform.AddTranslateOp().Set(translation)
        # print which translations were applied for x, y, z 
        print("X-Translation applied to object: " + str(transl_x))
        print("Y-Translation applied to object: " + str(transl_y))
        print("Z-Translation applied to object: " + str(transl_z))


        # Create rotation matrix from rotation_z_sim
        rot_z_sim = float(parts[14])    

        # Create rotation matrix via axis z for rotation_z_sim given in radian 
        cos_r = math.cos(rot_z_sim)
        sin_r = math.sin(rot_z_sim)

        rotation_z_mat = Gf.Matrix3d(
            cos_r, -sin_r, 0.0,
            sin_r,  cos_r, 0.0,
            0.0,     0.0,  1.0
        )
        rotation_mat = Gf.Matrix4d(rotation_z_mat, Gf.Vec3d(0.0, 0.0, 0.0))
        rotate_op = xform.AddTransformOp()
        rotate_op.Set(rotation_mat)
