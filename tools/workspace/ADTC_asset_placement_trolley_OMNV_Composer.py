import omni.kit.commands

import omni.usd
from pxr import Usd, UsdGeom, Gf,Sdf

import math
import os 

def load_empty_stage() -> Usd.Stage:
    """
    Loads an empty USD stage in Omniverse Composer and returns the stage object.

    Returns:
        Usd.Stage: The newly created empty stage, or None on failure.
    """
    try:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        print("Empty stage loaded successfully.")
        return stage
    except Exception as e:
        print(f"[ERROR] Could not load empty stage: {e}")
        return None

def set_default_prim(stage: Usd.Stage, prim: Usd.Prim):
    stage.SetDefaultPrim(prim)

def set_prim_world_pose(prim: Usd.Prim, translation: Gf.Vec3d):
    """
    Sets the world pose (translation only) for the given prim.
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()  # Clear existing ops if any existent
    xform.AddTranslateOp().Set(translation)

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

def automatic_asset_placement(sim_bbox_infos_path: str):
    # Load the USD stage
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("[ERROR] Failed to get USD stage.")
        return

    idx = 0

    with open(sim_bbox_infos_path, 'r') as f:
        lines = f.readlines()
        for bbox_sim in lines[1:]:  # Skip header
            parts = bbox_sim.strip().split(',')

            bbox_class = str(parts[0])
            asset_path = os.path.abspath(parts[-1].strip())

            if not os.path.exists(asset_path):
                print(f"[ERROR] Asset path does not exist: {asset_path}")
                continue
            print("Asset path:", asset_path)

            # Construct the prim path with unique identifier counting upwards for same class
            prim_path = f"/World/{bbox_class}_{idx}"
            print("Prim path:", prim_path)

            omni.kit.commands.execute(
                "CreatePayloadCommand",
                usd_context=omni.usd.get_context(),
                path_to=Sdf.Path(prim_path),
                asset_path=asset_path,
                instanceable=True
            )

            idx += 1

            prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
            if not prim.IsValid():
                print(f"[ERROR] Prim at path {prim_path} is not valid.")
                continue

            xform = UsdGeom.Xformable(prim)

            ## Apply the Translation ##
            # Get the transformation from bbox center to the prim's local coordinate frame
            translation_mat = get_Tr_bb_center_to_prim_cf(prim)

            transl_x = float(parts[11]) - translation_mat[3][0]
            transl_y = float(parts[12]) - translation_mat[3][1]
            transl_z = float(parts[13]) - translation_mat[3][2]

            translation = Gf.Vec3d(transl_x, transl_y, transl_z)
            xform.AddTranslateOp().Set(translation)
            print(f" Applied translation: x={transl_x:.2f}, y={transl_y:.2f}, z={transl_z:.2f}")

            # Create rotation matrix from rotation_z_sim
            rot_z_sim = float(parts[14])

            # Create rotation matrix via axis z for rotation_z_sim given in radians
            cos_r = math.cos(rot_z_sim)
            sin_r = math.sin(rot_z_sim)

            rotation_z_mat = Gf.Matrix3d(
                cos_r, -sin_r, 0.0,
                sin_r,  cos_r, 0.0,
                0.0,     0.0,  1.0
            )
            rotation_mat = Gf.Matrix4d(rotation_z_mat, Gf.Vec3d(0.0, 0.0, 0.0))
            xform.AddTransformOp().Set(rotation_mat)


def main(sim_bbox_infos_path: str = None): 

    # world = World(stage_units_in_meters=1.0)                           ##### set the unit of the world to meters respectively cm respectively mm
    
    stage = load_empty_stage()

    if stage is None:
        print("Failed to load empty stage.")

    default_prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    set_default_prim(stage, default_prim)

    omni.usd.get_context().get_stage().GetRootLayer().Reload()
    omni.kit.commands.execute("CreateGroundPlaneCommand", xform_path="/World/GroundPlane")
    omni.kit.commands.execute("CreateDistantLightCommand", path="/World/Camera/CameraLight")

    automatic_asset_placement(sim_bbox_infos_path)


if __name__ == "__main__":
    # Path to be adapted
    sim_bbox_infos_path = '/home/leo/workspace/Omniverse/2025-04-12_09-16-18_Inference_ImageSet_predicted bboxes_SIM_world_cf_POST_PROCESSED.csv'
    
    main(sim_bbox_infos_path)
    
