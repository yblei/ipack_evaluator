import sys
sys.path.append('/home/blei/BagBuddy/evaluation/ipack_evaluator/3D-bin-packing')

from py3dbp import Packer, Bin, Item, Painter
from ipack_evaluator.packing_eval import run_exp
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless
import matplotlib.pyplot as plt
from pathlib import Path


def run_jerry_3dbp_evaluation(grocery_items, boxes, visualizer):
    """
    Uses the jerry 3D-bin-packing algorithm to pack grocery items into boxes.
    Returns the grocery items with obb_target set.
    """
    # Initialize packer
    packer = Packer()
    
    # Add bins (boxes) to packer
    # boxes is a dict: {'0x0x0': array([width, height, depth])}
    # Convert from meters to millimeters to avoid numerical issues
    SCALE = 1000  # meters to millimeters
    
    for box_name, box_dims in boxes.items():
        width, height, depth = box_dims * SCALE  # convert to mm
        max_weight = 999  # default max weight
        
        bin_obj = Bin(
            partno=box_name,
            WHD=(float(width), float(height), float(depth)),
            max_weight=max_weight,
            corner=0,
            put_type=1  # open top packing order
        )
        packer.addBin(bin_obj)
    
    # Add items to packer
    for item in grocery_items:
        # Extract item properties from GroceryItem
        # Convert from meters to millimeters
        width, height, depth = item.extent * SCALE  # convert to mm
        weight = 1  # default weight
        
        item_obj = Item(
            partno=item.name,
            name=item.name,
            typeof='cube',  # assuming cube shape
            WHD=(float(width), float(height), float(depth)),
            weight=weight,
            level=1,  # packing priority (lower = higher priority)
            loadbear=100,  # load bearing capacity
            updown=True,  # can be placed upside down
            color='blue'
        )
        packer.addItem(item_obj)
    
    # Run packing algorithm
    packer.pack(
        bigger_first=True,
        distribute_items=True,
        fix_point=True,  # fix floating items
        check_stable=True,  # check stability
        support_surface_ratio=0.75,
        number_of_decimals=3
    )
    
    # Visualize and save packing results
    output_dir = Path("/home/blei/BagBuddy/evaluation/ipack_evaluator/experiment_output/jerry_3dbp_0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, bin_obj in enumerate(packer.bins):
        painter = Painter(bin_obj)
        fig = painter.plotBoxAndItems(
            title=bin_obj.partno,
            alpha=0.2,
            write_num=True,
            fontsize=10
        )
        # Save the figure
        output_path = output_dir / f"packing_bin_{idx}_{bin_obj.partno}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        #plt.close(fig)
        print(f"Saved packing visualization to {output_path}")
    
    # Map results back to grocery_items with obb_target
    for bin_obj in packer.bins:
        for packed_item in bin_obj.items:
            # Find corresponding grocery item by name
            grocery_item = next(
                (gi for gi in grocery_items 
                 if gi.name == packed_item.partno),
                None
            )
            
            if grocery_item:
                # Convert position back from mm to meters
                position_m = np.array([float(p) / SCALE for p in 
                                       packed_item.position])
                
                # Get dimensions after rotation (convert Decimal to float)
                dims_m = np.array([float(d) / SCALE for d in 
                                   packed_item.getDimension()])
                
                # Get original dimensions
                original_dims_m = grocery_item.extent  # from grocery_item
                
                # Get rotation matrix
                R = get_rotation_matrix_from_type(
                    packed_item.rotation_type,
                    original_dims_m,
                    dims_m
                )

                # add extend/2 to grocery item
                position_m += dims_m / 2

                position_m = list(position_m)
                dims_m = list(dims_m)
                
                # Create Open3D OrientedBoundingBox for obb_target
                grocery_item.obb_target = o3d.geometry.OrientedBoundingBox(
                    center=position_m,
                    R=R,  # Actual rotation matrix!
                    extent=dims_m
                )
    
    # Handle unfitted items
    for unfitted in packer.unfit_items:
        print("Unfitted item:", unfitted.partno)
        raise RuntimeError(
            f"Packing failed, item {unfitted.partno} could not be packed."
        )
        grocery_item = next(
            (gi for gi in grocery_items 
             if gi.name == unfitted.partno),
            None
        )
        if grocery_item:
            grocery_item.obb_target = None  # Mark as unable to pack
    
    return grocery_items


def get_rotation_matrix_from_type(rotation_type, original_whd, rotated_whd):
    """
    Convert jerry rotation_type to a rotation matrix.
    
    Jerry's rotation_type (0-5) represents which orientation was chosen.
    We derive the rotation by comparing original WHD with rotated WHD.
    
    Args:
        rotation_type: int 0-5 from jerry
        original_whd: tuple (W, H, D) original dimensions
        rotated_whd: tuple (W, H, D) after rotation from getDimension()
    
    Returns:
        3x3 rotation matrix (proper rotation, det=+1)
    """
    from scipy.spatial.transform import Rotation as R
    
    # Map which axis became which
    orig = np.array(original_whd)
    rotated = np.array(rotated_whd)
    
    # Define the 6 possible proper rotations for a box
    # These are rotations that align box axes with coordinate axes
    rotation_matrices = [
        np.eye(3),                                      # 0: No rotation (WHD -> WHD)
        R.from_euler('z', 90, degrees=True).as_matrix(),   # 1: 90° around Z
        R.from_euler('z', 180, degrees=True).as_matrix(),  # 2: 180° around Z
        R.from_euler('z', 270, degrees=True).as_matrix(),  # 3: 270° around Z
        R.from_euler('x', 90, degrees=True).as_matrix(),   # 4: 90° around X
        R.from_euler('y', 90, degrees=True).as_matrix(),   # 5: 90° around Y
    ]
    
    # Try to find which rotation matrix produces the rotated dimensions
    for rot_matrix in rotation_matrices:
        # Apply rotation and check if dimensions match
        rotated_test = np.abs(rot_matrix @ orig)  # Use abs because we only care about dimensions
        if np.allclose(rotated_test, rotated, atol=1e-4):
            # Verify it's a proper rotation (det = +1)
            if np.isclose(np.linalg.det(rot_matrix), 1.0):
                return rot_matrix
    
    # If no exact match, find the permutation and ensure proper rotation
    # Check all permutations and their dimension matches
    permutations = [
        (0, 1, 2),  # WHD -> WHD
        (0, 2, 1),  # WHD -> WDH
        (1, 0, 2),  # WHD -> HWD
        (1, 2, 0),  # WHD -> HDW
        (2, 0, 1),  # WHD -> DWH
        (2, 1, 0),  # WHD -> DHW
    ]
    
    for perm in permutations:
        if np.allclose(rotated, orig[list(perm)], atol=1e-4):
            # Found matching permutation
            # Create permutation matrix
            P = np.zeros((3, 3))
            for i, j in enumerate(perm):
                P[i, j] = 1
            
            # Check determinant
            det = np.linalg.det(P)
            
            if np.isclose(det, 1.0):
                # Already a proper rotation
                return P
            elif np.isclose(det, -1.0):
                # It's a reflection, we need to fix it
                # Flip one axis to make it a proper rotation
                # Just use identity instead since we can't represent this reflection
                print(f"Warning: Rotation {rotation_type} involves reflection, using identity")
                return np.eye(3)
    
    # Fallback to identity if no match found
    print(f"Warning: Could not determine rotation for type {rotation_type}, using identity")
    return np.eye(3)


if __name__ == "__main__":
    exp_name = "jerry_3dbp"
    run_exp(run_jerry_3dbp_evaluation, exp_name=exp_name, enable_gui=False, static=True)
