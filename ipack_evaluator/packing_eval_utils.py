"""Utility functions for packing evaluation, including footprint generation."""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion, zoom
from ipack_evaluator.grocery_item import GroceryItem
import pandas as pd


def get_footprint(
    grocery_item,
    final_pose: dict,
    box_size: tuple = (0.35, 0.25, 0.1),
    grid_resolution: float = 0.005,
) -> np.ndarray:
    """
    Generate a 2D occupancy grid (footprint) for a grocery item in its final
    pose.

    Args:
        grocery_item: GroceryItem with extent and mesh_path
        final_pose: Dict with 'position' and 'quaternion' [w,x,y,z]
        box_size: (width, height, depth) of the container in meters
        grid_resolution: Size of each grid cell in meters

    Returns:
        2D numpy array representing occupancy grid (1=occupied, 0=free)
    """
    print(f"Attempting mesh sampling for {grocery_item.name}")

    # Try mesh sampling first
    try:
        footprint = _get_footprint_from_mesh(
            grocery_item, final_pose, box_size, grid_resolution
        )
        if footprint is not None:
            return footprint
    except Exception as e:
        raise RuntimeError("Running without trimesh support is not supported.") from e
        print(
            f"Warning: Mesh sampling failed for {grocery_item.name}: {e}. "
            "Falling back to bounding box."
        )

    # Fallback to bounding box sampling
    print(f"Using bounding box sampling for {grocery_item.name}")
    return _get_footprint_from_bbox(
        grocery_item, final_pose, box_size, grid_resolution
    )


def _get_footprint_from_mesh(
    grocery_item, final_pose: dict, box_size: tuple, grid_resolution: float
) -> np.ndarray:
    """Sample points from the actual mesh geometry."""
    try:
        import trimesh
    except ImportError:
        print(
            f"Warning: trimesh not available for {grocery_item.name}. "
            "Falling back to bbox."
        )
        raise ImportError("trimesh not available")

    # Load the mesh
    mesh_path = grocery_item.sim_spec.mesh_path
    print(f"Loading mesh from: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path)
        print(
            f"Mesh loaded successfully. Vertices: {len(mesh.vertices)}, "
            f"Faces: {len(mesh.faces)}"
        )
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        raise

    # Sample points from mesh
    try:
        if hasattr(mesh, "is_watertight") and mesh.is_watertight:
            # Sample from volume if mesh is watertight
            num_samples = max(1000, int(np.prod(grocery_item.extent) * 50000))
            points = mesh.sample_volume(num_samples)
            print(f"Sampling {num_samples} points from volume")
        else:
            # Sample from surface
            num_samples = max(500, int(np.sum(grocery_item.extent) * 10000))
            sample_result = mesh.sample(num_samples)
            if isinstance(sample_result, tuple) and len(sample_result) == 2:
                points, face_indices = sample_result
            else:
                points = sample_result
            print(f"Sampling {num_samples} points from surface")
    except Exception as e:
        print(f"Error during mesh sampling: {e}")
        # Fallback to vertex sampling
        max_vertices = len(mesh.vertices)
        min_samples = max(500, max_vertices // 10)
        num_samples = min(max_vertices, min_samples)
        indices = np.random.choice(max_vertices, num_samples, replace=False)
        points = mesh.vertices[indices]
        print(f"Using {num_samples} mesh vertices as fallback")

    # Convert from centimeters to meters (mesh models are in cm)
    points = points * 0.01
    print(f"Sampled points shape: {points.shape}")
    print(
        f"After cm->m conversion, points range: "
        f"x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
        f"y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
        f"z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]"
    )

    return _project_points_to_grid(
        points, final_pose, box_size, grid_resolution
    )


def _get_footprint_from_bbox(
    grocery_item, final_pose: dict, box_size: tuple, grid_resolution: float
) -> np.ndarray:
    """Generate footprint from oriented bounding box sampling."""
    # Create oriented bounding box
    center = np.array([0, 0, 0])
    extent = np.array(grocery_item.extent)

    obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extent)

    # Sample points from the bounding box
    num_points = int(np.prod(extent) * 1000000)  # Density-based sampling
    # Clamp to reasonable range
    num_points = max(1000, min(num_points, 200000))

    points = []
    for _ in range(num_points):
        # Sample random point in unit cube [-0.5, 0.5]^3
        rand_point = np.random.uniform(-0.5, 0.5, 3)
        # Scale by extent to get point in bounding box
        scaled_point = rand_point * extent
        points.append(scaled_point)

    points = np.array(points)

    return _project_points_to_grid(
        points, final_pose, box_size, grid_resolution
    )


def _project_points_to_grid(
    points: np.ndarray,
    final_pose: dict,
    box_size: tuple,
    grid_resolution: float,
) -> np.ndarray:
    """Project 3D points to a 2D occupancy grid after pose transformation."""
    # Extract pose information
    position = np.array(final_pose["position"])
    quaternion = np.array(final_pose["quaternion"])  # [w, x, y, z] format

    # print(f"Position: {position}")
    # print(f"Quaternion: {quaternion}")
    # print(f"Box size: {box_size}")

    # Convert quaternion to rotation matrix
    # MuJoCo uses [w, x, y, z] format
    rotation = Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    )
    rotation_matrix = rotation.as_matrix()

    # Transform points to world coordinates
    world_points = points @ rotation_matrix.T + position

    # print(f"Input points shape: {points.shape}")
    # print(f"World points range: x=[{world_points[:, 0].min():.3f}, "
    #      f"{world_points[:, 0].max():.3f}], "
    #      f"y=[{world_points[:, 1].min():.3f}, "
    #      f"{world_points[:, 1].max():.3f}]")

    # Create grid
    width, height = box_size[0], box_size[1]  # x, y dimensions
    grid_width = int(width / grid_resolution)
    grid_height = int(height / grid_resolution)

    # print(f"Grid dimensions: {grid_width} x {grid_height}")

    # Project to 2D and convert to grid indices
    # Origin is at bottom-left (0,0), so no coordinate flipping needed
    x_coords = world_points[:, 0]  # x coordinates
    y_coords = world_points[:, 1]  # y coordinates

    # Convert to grid indices
    grid_x = (x_coords / grid_resolution).astype(int)
    grid_y = (y_coords / grid_resolution).astype(int)

    # Filter points within bounds
    valid_mask = (
        (grid_x >= 0)
        & (grid_x < grid_width)
        & (grid_y >= 0)
        & (grid_y < grid_height)
    )

    valid_grid_x = grid_x[valid_mask]
    valid_grid_y = grid_y[valid_mask]

    # print(f"Points in bounds: {np.sum(valid_mask)}/{len(points)}")

    # Create occupancy grid
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Mark occupied cells
    if len(valid_grid_x) > 0:
        grid[valid_grid_y, valid_grid_x] = 1

    occupied_cells = np.sum(grid)
    print(f"Grid cells occupied: {occupied_cells}")

    return grid


def get_density(grocery_items, box_size):
    """
    Calculate the percentage of occupied space inside the package.

    Args:
        grocery_items: List of GroceryItem objects with SimSpec containing
                       volume
        box_size: (width, depth, height) box dimensions in meters

    Returns:
        float: Percentage of occupied space (0-100)
    """
    # Calculate total box volume in cubic meters
    box_volume = box_size[0] * box_size[1] * box_size[2]

    # Sum up volumes of all grocery items
    total_item_volume = 0.0
    for item in grocery_items:
        if item.sim_spec is not None and item.sim_spec.volume is not None:
            total_item_volume += item.sim_spec.volume
        else:
            raise ValueError(
                f"No volume data available for item {item.name}. "
                f"Ensure SimSpec.volume is calculated."
            )

    # Calculate density as percentage
    if box_volume <= 0:
        raise ValueError("Box volume must be positive")

    density_percentage = (total_item_volume / box_volume) * 100

    print(f"Total item volume: {total_item_volume:.6f} m³")
    print(f"Box volume: {box_volume:.6f} m³")
    print(f"Density: {density_percentage:.1f}%")

    density_fraction = density_percentage / 100.0

    return density_fraction


def get_overlaps(items: list[GroceryItem], overlap_threshold: int = 15) -> list[tuple[str, str]]:
    """Check for overlaps between grocery items based on their footprints.

    Args:
        items (list[GroceryItem]): List of grocery items to check for overlaps.

    Returns:
        list[tuple[str, str]]: List of tuples containing pairs of overlapping
                               item names.
    """

    # sort by z axis position (highest to lowest)
    items = sorted(
        items,
        key=lambda item: item.sim_spec.final_pose["position"][2],
        reverse=True,
    )

    # get a unique set of item names
    item_names = sorted(set([item.name for item in items]))

    # get a dataframe with names at x and y axis
    overlap_df = pd.DataFrame(
        0, index=item_names, columns=item_names, dtype=int
    )

    # Compare each item to all items below it
    for i, current_item in enumerate(items):
        # Get footprint for current item from pre-computed SimSpec
        current_footprint = current_item.sim_spec.footprint

        # Compare to all items below (j > i means lower z position)
        for j, below_item in enumerate(items[i + 1:], start=i + 1):
            # Get footprint for item below from pre-computed SimSpec
            below_footprint = below_item.sim_spec.footprint

            # Upsample both footprints by a factor of 2, then erode by 1 pixel
            # This gives us finer control - equivalent to 0.5 pixel erosion
            # at original resolution
            current_upsampled = zoom(
                current_footprint.astype(float), 2, order=0
            )
            below_upsampled = zoom(below_footprint.astype(float), 2, order=0)
            
            # Convert back to binary and erode by 1 pixel
            current_footprint_processed = binary_erosion(
                current_upsampled > 0.5
            )
            below_footprint_processed = binary_erosion(below_upsampled > 0.5)

            # Check for overlap by element-wise multiplication
            # If both footprints have 1 at same position, there's overlap
            overlap_mask = (
                current_footprint_processed * below_footprint_processed
            )
            
            # Count the number of overlapping pixels and require at least 3
            overlap_pixel_count = np.sum(overlap_mask)
            has_overlap = overlap_pixel_count >= overlap_threshold

            if has_overlap:
                # Increment overlap count in dataframe
                overlap_df.loc[current_item.name, below_item.name] += 1

    return overlap_df
