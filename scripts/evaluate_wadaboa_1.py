import sys

sys.path.append("/home/blei/BagBuddy/evaluation/ipack_evaluator/3d-bpp/src")

from ipack_evaluator.packing_eval import run_exp
import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path

# Import wadaboa modules
import main as wadaboa_main
import utils
import config as wadaboa_config
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R_scipy


def run_wadaboa_baseline_evaluation(grocery_items, boxes, visualizer, mode: str = "bl"):
    """
    Uses the Wadaboa baseline (bl) algorithm to pack grocery items into boxes.
    Returns the grocery items with obb_target set.
    """
    # Convert from meters to millimeters (Wadaboa uses mm)
    SCALE = 1000

    # Get box dimensions (assumes single box)
    assert len(boxes) == 1, "Only one box size should be provided"
    box_name, box_dims = list(boxes.items())[0]
    box_width, box_height, box_depth = box_dims * SCALE

    # Create pallet dimensions for Wadaboa
    pallet_dims = utils.Dimension(
        width=int(box_width),
        depth=int(box_depth),
        height=int(box_height),
        weight=999999,  # Large weight capacity
    )

    # Convert grocery_items to Wadaboa order format (pandas DataFrame)
    # Expected columns: width, depth, height, weight, quantity
    items_data = []
    item_name_map = {}  # Map index to grocery item

    for idx, item in enumerate(grocery_items):
        width, height, depth = item.extent * SCALE
        items_data.append(
            {
                "width": int(width),
                "depth": int(depth),
                "height": int(height),
                "weight": 1,  # Default weight
                "quantity": 1,  # Each grocery item is unique
            }
        )
        item_name_map[idx] = item

    order = pd.DataFrame(items_data)

    # Override config with our pallet dimensions
    wadaboa_config.PALLET_DIMS = pallet_dims

    # Run the MaxRects procedure (avoiding OR-Tools compatibility issues)
    print(f"Running Wadaboa MaxRects with {len(order)} items...")
    try:
        bin_pool = wadaboa_main.main(
            order,
            procedure=mode,  # MaxRects instead of baseline
            tlim=60,  # Time limit in seconds
            enable_solver_output=False,
        )
    except KeyError as e:
        raise RuntimeError(f"Wadaboa packing failed with error: {e}")

    print(f"Packing completed: {len(bin_pool.compact_bins)} bins used")

    # Visualize and save packing results
    output_dir = Path("/home/blei/BagBuddy/evaluation/ipack_evaluator/experiment_output") / f"wadaboa_{mode}_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the dataframe for visualization and mapping
    bin_df = bin_pool.to_dataframe()
    
    # Create visualization for each bin
    for bin_idx in bin_df['bin'].unique():
        bin_data = bin_df[bin_df['bin'] == bin_idx]
        output_path = output_dir / f"packing_bin_{bin_idx}.png"
        plot_wadaboa_packing(bin_data, pallet_dims, output_path)
        print(f"Saved packing visualization to {output_path}")


    if len(bin_df["bin"].unique()) != 1:
        raise RuntimeError(f"Packing failed, used {len(bin_df['bin'].unique())} bins instead of 1.")

    #raise NotADirectoryError("It seems like the pipeline does not raise an error when not all object fit. Maybe make sure the lines in the dataframe is equal to the number of grocery items?")

    # Map results back to grocery_items with obb_target



    for idx, row in bin_df.iterrows():
        # row contains: item (index), x, y, z, width, depth, height, etc.
        product_idx = row["item"]

        if product_idx in item_name_map:
            grocery_item = item_name_map[product_idx]

            # Convert position from mm back to meters
            # Position is bottom-left-back corner, convert to center
            position_m = np.array(
                [
                    (row["x"] + row["width"] / 2) / SCALE,
                    (row["y"] + row["depth"] / 2) / SCALE,
                    (row["z"] + row["height"] / 2) / SCALE,
                ]
            )

            # Convert dimensions back to meters
            dims_m = np.array(
                [row["width"] / SCALE, row["depth"] / SCALE, row["height"] / SCALE]
            )
            R = infer_rotation_matrix(
                grocery_item.extent, dims_m, SCALE
            )
            # Create Open3D OrientedBoundingBox for obb_target
            grocery_item.obb_target = o3d.geometry.OrientedBoundingBox(
                center=position_m, R=R, extent=dims_m
            )

    # Check for unfitted items
    unfitted_items = [item for item in grocery_items if item.obb_target is None]

    if unfitted_items:
        unfitted_names = [item.name for item in unfitted_items]
        print(
            f"Warning: {len(unfitted_items)} items could not be packed: {unfitted_names}"
        )
        raise RuntimeError(
            f"Packing failed, {len(unfitted_items)} items could not be packed."
        )


    if bin_df.shape[0] != len(grocery_items):
        raise RuntimeError(
            f"Packing failed, only {bin_df.shape[0]} out of {len(grocery_items)} items were packed."
        )
    return grocery_items

def infer_rotation_matrix(original_dims, packed_dims, SCALE):
    """
    Infers a proper rotation matrix by comparing dimensions.
    Returns a 3x3 rotation matrix with det = +1.
    """
    # Original dimensions in mm
    orig = np.array(original_dims)# * SCALE
    packed = np.array(packed_dims)
    
    # Try common rotations that Wadaboa might apply
    rotation_tests = [
        (np.eye(3), "no rotation"),
        (R_scipy.from_euler('z', 90, degrees=True).as_matrix(), "90° Z"),
        (R_scipy.from_euler('z', 180, degrees=True).as_matrix(), "180° Z"),
        (R_scipy.from_euler('z', 270, degrees=True).as_matrix(), "270° Z"),
        (R_scipy.from_euler('x', 90, degrees=True).as_matrix(), "90° X"),
        (R_scipy.from_euler('y', 90, degrees=True).as_matrix(), "90° Y"),
    ]
    
    for R_test, desc in rotation_tests:
        # Apply rotation to original dimensions (use abs for dimension comparison)
        rotated = np.abs(R_test @ orig)
        if np.allclose(rotated, packed, atol=0.002):  # 2mm tolerance
            return R_test
    
    # Fallback: identity
    print(f"Warning: Could not match rotation. Orig: {orig}, Packed: {packed}")
    return np.eye(3)

def plot_wadaboa_packing(bin_df, pallet_dims, output_path):
    """Plot the Wadaboa packing results"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the bin/pallet outline
    ax.set_xlim([0, pallet_dims.width])
    ax.set_ylim([0, pallet_dims.depth])
    ax.set_zlim([0, pallet_dims.height])

    # Plot each item as a box
    for idx, row in bin_df.iterrows():
        x, y, z = row["x"], row["y"], row["z"]
        w, d, h = row["width"], row["depth"], row["height"]

        # Create box vertices
        vertices = [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + d, z],
            [x, y + d, z],  # bottom
            [x, y, z + h],
            [x + w, y, z + h],
            [x + w, y + d, z + h],
            [x, y + d, z + h],  # top
        ]

        # Define the 6 faces
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
        ]

        # Add to plot
        collection = Poly3DCollection(faces, alpha=0.3, edgecolor="k")
        ax.add_collection3d(collection)

    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_zlabel("Height (mm)")
    ax.set_title("Wadaboa Packing")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    mode = "cg" # "mr" for MaxRects, "cg" for Column Generation, "bl" for baseline
    exp_name = f"wadaboa_{mode}"

    run_exp(
        partial(run_wadaboa_baseline_evaluation, mode=mode),
        exp_name=exp_name,
        enable_gui=False,
        static=True,
    )
