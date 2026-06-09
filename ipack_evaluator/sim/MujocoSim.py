# flake8: noqa

import mujoco  # MuJoCo Python bindings
import numpy as np
import time
from mujoco import viewer, mj_step
import open3d as o3d
from ipack_evaluator.grocery_item import GroceryItem, SimSpec
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2

import logging
logger = logging.getLogger(__name__)

class MujocoSim:
    def __init__(self, visualizer=None):
        self.asset_lines = []
        self.body_lines = []
        self.box_extent = None
        self.box_alpha = 0.5
        self.visualizer = visualizer

    def item_to_name(self, item: GroceryItem) -> str:
        """Get a valid MuJoCo name for a GroceryItem."""
        return item.name

    def add_hope_grocery_item(self, item: GroceryItem):
        """Add a grocery item to the simulation."""

        assert (
            item.obb_target is not None
        ), "GroceryItem must have an obb_target to be added to the simulation."

        name = self.item_to_name(item)
        obj_path = item.sim_spec.mesh_path
        mat_name = f"mat_{name}"
        mesh_name = f"mesh_{name}"

        # Extract position from Open3D OrientedBoundingBox
        x, y, z = item.obb_target.center

        # Mujoco uses z-up, while Open3D uses y-up
        # We need to convert the rotation matrix accordingly
        rot_matrix = np.array(item.obb_target.R)
        # rotate around y axis by 90deg
        rot_y_90 = R.from_euler("y", 90, degrees=True).as_matrix()
        # rot_matrix = rot_y_90 @ rot_matrix
        import_rotation = item.sim_spec.import_transform[:3, :3]
        rotation = import_rotation @ rot_matrix

        # Convert rotation matrix to quaternion using scipy
        quat = R.from_matrix(rotation).as_quat()  # Returns [x, y, z, w]

        # MuJoCo expects quaternion as [w, x, y, z]
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

        obj_dir = Path(obj_path).parent
        texture_file = (obj_dir / "texture_map.png").as_posix()

        self.asset_lines.append(
            f'<texture name="tex_{name}" type="2d" file="{texture_file}"/>'
        )
        self.asset_lines.append(
            f'<material name="{mat_name}" texture="tex_{name}" specular="0.3" shininess="0.7" />'
        )
        self.asset_lines.append(
            f'<mesh name="{mesh_name}" file="{obj_path}" scale="0.01 0.01 0.01" />'
        )

        self.body_lines.append(
            f"""
            <body name="{name}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
                <freejoint/>
                <geom type="mesh" mesh="{mesh_name}" material="{mat_name}" rgba="1 1 1 1" 
                      mass="0.5" friction="0.8 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
            </body>"""
        )

    def set_box(self, box_extent: np.ndarray, box_alpha=0.5):
        """
        Set the bounding box with given extent and alpha.
        Origin is at the front-left corner of the box.
        X goes along the width, Y along the depth, Z along the height.

        Args:
            box_extent (np.array): The size of the box in x, y, z dimensions.
            box_alpha (float): The transparency of the box walls.
        """
        assert isinstance(box_extent, np.ndarray)
        self.box_extent = box_extent  # Store box extent for camera positioning
        box_x, box_y, box_z = box_extent
        wall_thickness_visual = 0.003  # 3 mm for visual
        wall_thickness_collision = 0.02  # 2 cm for collision

        # Box center offset from origin (front-left corner)
        box_center_x = box_x / 2
        box_center_y = box_y / 2
        box_center_z = box_z / 2

        box_material = (
            f'<material name="cardboard" rgba="0.8 0.6 0.3 {box_alpha}" />'
        )

        box_geoms = f"""
            <!-- Visual geoms (thin, group 0 = visible) -->
            <geom name="visual_bottom" group="0" type="box" material="cardboard" pos="{box_center_x} {box_center_y} {wall_thickness_visual/2}" size="{box_x/2} {box_y/2} {wall_thickness_visual/2}" contype="0" conaffinity="0" />
            <geom name="visual_left" group="0" type="box" material="cardboard" pos="{wall_thickness_visual/2} {box_center_y} {box_center_z}" size="{wall_thickness_visual/2} {box_y/2} {box_z/2}" contype="0" conaffinity="0" />
            <geom name="visual_right" group="0" type="box" material="cardboard" pos="{box_x-wall_thickness_visual/2} {box_center_y} {box_center_z}" size="{wall_thickness_visual/2} {box_y/2} {box_z/2}" contype="0" conaffinity="0" />
            <geom name="visual_front" group="0" type="box" material="cardboard" pos="{box_center_x} {wall_thickness_visual/2} {box_center_z}" size="{box_x/2} {wall_thickness_visual/2} {box_z/2}" contype="0" conaffinity="0" />
            <geom name="visual_back" group="0" type="box" material="cardboard" pos="{box_center_x} {box_y-wall_thickness_visual/2} {box_center_z}" size="{box_x/2} {wall_thickness_visual/2} {box_z/2}" contype="0" conaffinity="0" />
            
            <!-- Collision geoms (thick, group 3 = invisible, extending outward) -->
            <geom name="collision_bottom" group="3" type="box" pos="{box_center_x} {box_center_y} {-wall_thickness_collision + wall_thickness_visual/2}" size="{box_x/2 + wall_thickness_collision} {box_y/2 + wall_thickness_collision} {wall_thickness_collision}" contype="1" conaffinity="1" />
            <geom name="collision_left" group="3" type="box" pos="{-wall_thickness_collision + wall_thickness_visual/2} {box_center_y} {box_center_z}" size="{wall_thickness_collision} {box_y/2 + wall_thickness_collision} {box_z/2}" contype="1" conaffinity="1" />
            <geom name="collision_right" group="3" type="box" pos="{box_x + wall_thickness_collision - wall_thickness_visual/2} {box_center_y} {box_center_z}" size="{wall_thickness_collision} {box_y/2 + wall_thickness_collision} {box_z/2}" contype="1" conaffinity="1" />
            <geom name="collision_front" group="3" type="box" pos="{box_center_x} {-wall_thickness_collision + wall_thickness_visual/2} {box_center_z}" size="{box_x/2 + wall_thickness_collision} {wall_thickness_collision} {box_z/2}" contype="1" conaffinity="1" />
            <geom name="collision_back" group="3" type="box" pos="{box_center_x} {box_y + wall_thickness_collision - wall_thickness_visual/2} {box_center_z}" size="{box_x/2 + wall_thickness_collision} {wall_thickness_collision} {box_z/2}" contype="1" conaffinity="1" />
        """

        self.box_material = box_material
        self.box_geoms = box_geoms

    def _extract_body_poses(self, m, d):
        """Extract poses of all bodies in the simulation."""
        poses = {}

        # Iterate through all bodies (skip world body at index 0)
        for body_id in range(1, m.nbody):
            body_name = m.body(body_id).name
            if not body_name:
                continue

            # Get position and quaternion from simulation data
            pos = d.xpos[body_id].copy()
            quat = d.xquat[body_id].copy()  # [w, x, y, z] format

            poses[body_name] = {"position": pos, "quaternion": quat}

        return poses

    def _generate_mjcf(self, gravity=(0, 0, 0)):
        """Generate MJCF XML string for the simulation.

        Args:
            gravity (tuple): Gravity vector (x, y, z). Default is (0, 0, 0).
        """

        # check if box is set
        assert self.box_geoms is not None, "Box must be set using set_box()"

        asset = "\n        ".join(self.asset_lines)
        bodies = "\n        ".join(self.body_lines)
        gravity_str = f"{gravity[0]} {gravity[1]} {gravity[2]}"

        mjcf = f"""<mujoco model="grocery_scene">
                    <compiler angle="degree" coordinate="local"/>
                    <option gravity="{gravity_str}" solver="Newton" iterations="50" timestep="0.001"/>
                    <visual>
                        <global offwidth="2048" offheight="1536"/>
                    </visual>
                    <default>
                        <geom solimp="0.95 0.95 0.01" solref="0.02 1" friction="0.7 0.005 0.0001"/>
                        <joint damping="0.1" frictionloss="0.01"/>
                    </default>
                    <asset>
                        {asset}
                        {self.box_material}
                    </asset>
                    <worldbody>
                        <light name="main_light" pos="0 0 200" dir="0 0 -1" diffuse="1 1 1" specular=".5 .5 .5" directional="true" castshadow="true" />
                        <body name="cardboard_box" pos="0 0 0">
                            {self.box_geoms}
                        </body>
                        {bodies}
                    </worldbody>
                </mujoco>"""
        return mjcf

    def run_sim(
        self,
        fps=30,
        speed=1.0,
        static=False,
        settle_time=2.0,
        max_sim_time=None,
        use_gui=True,
    ):
        """
        Run the MuJoCo simulation with viewer.

        Args:
            fps (int): Frames per second for the simulation.
            speed (float): Speed multiplier for the simulation.
            static (bool): If True, the simulation will not step forward
                (static view).
            settle_time (float): Time to let objects settle before viewer starts.
            max_sim_time (float): Maximum simulation time in seconds. If None, run indefinitely.

        Returns:
            dict: Final poses of all bodies in the simulation. Keys are body names,
                  values are dicts with 'position' and 'quaternion' keys.
        """
        gravity = (0, 0, -9.81)  # Gravity vector
        mjcf = self._generate_mjcf(gravity=gravity)
        m = mujoco.MjModel.from_xml_string(mjcf)
        d = mujoco.MjData(m)
        n = len(self.body_lines)

        if use_gui:
            with viewer.launch_passive(m, d) as v:
                # Calculate the actual center of the box dynamically
                if self.box_extent is not None:
                    box_center_x = self.box_extent[0] / 2
                    box_center_y = self.box_extent[1] / 2
                    box_center_z = self.box_extent[2] / 2
                else:
                    # Fallback to default if box_extent not set
                    box_center_x, box_center_y, box_center_z = 0.1, 0.15, 0.1

                v.cam.azimuth = 0  # Doesn't matter when looking straight down
                v.cam.elevation = -90  # Look straight down
                v.cam.lookat[:] = [box_center_x, box_center_y, box_center_z]
                v.cam.distance = (
                    0.8  # Increased distance for better top-down view
                )

                dt = 1.0 / fps / speed
                sim_start_time = d.time if not static else 0

                while v.is_running():
                    if not static:
                        mj_step(m, d)
                        # Check if we've exceeded max simulation time
                        if (
                            max_sim_time is not None
                            and (d.time - sim_start_time) >= max_sim_time
                        ):
                            print(
                                f"Simulation completed after {d.time - sim_start_time:.2f} seconds of sim time"
                            )
                            break
                        if d.time - sim_start_time >= settle_time:
                            break

                        print(f"Sim time: {d.time:.3f}s", end="\r")
                    v.sync()
                    time.sleep(dt)

        else:
            if not static and settle_time > 0:
                # Temporarily use larger timestep for faster settling
                original_timestep = m.opt.timestep
                m.opt.timestep = 0.005  # 5ms timestep for settling (5x faster)

                settle_steps = int(settle_time / m.opt.timestep)
                print(
                    f"Settling objects for {settle_time}s ({settle_steps} steps at {m.opt.timestep}s timestep)..."
                )

                # Add progress reporting every 100 steps (since fewer total steps now)
                for step in range(settle_steps):
                    if step % 100 == 0 and step > 0:
                        elapsed_sim_time = step * m.opt.timestep
                        print(
                            f"  Settlement progress: {elapsed_sim_time:.2f}s / {settle_time:.2f}s ({step}/{settle_steps} steps)"
                        )
                    mj_step(m, d)
            else:
                #raise ValueError(
                 #   "Static simulation without GUI is not supported."
                #)
                mj_step(m, d) # single step for static view

        # Render and save the final frame after simulation completes
        if self.visualizer is not None:
            # Use MuJoCo's renderer for offscreen rendering at Full HD resolution
            renderer = mujoco.Renderer(m, height=1080, width=1920)

            # Update scene without specifying camera (uses default)
            renderer.update_scene(d)

            # Set up camera for top-down view manually
            # Calculate the actual center of the box dynamically
            if self.box_extent is not None:
                box_center_x = self.box_extent[0] / 2
                box_center_y = self.box_extent[1] / 2
                box_center_z = self.box_extent[2] / 2
            else:
                # Fallback to default if box_extent not set
                box_center_x, box_center_y, box_center_z = 0.1, 0.15, 0.1

            # Render the scene using default camera settings
            # The renderer will use a reasonable default viewpoint
            rendered_image = renderer.render()

            # Convert to BGR for OpenCV (if needed)
            if rendered_image.shape[2] == 3:  # RGB
                rendered_image = cv2.cvtColor(
                    rendered_image, cv2.COLOR_RGB2BGR
                )

            # Save using the visualizer
            self.visualizer.save_sim_box_render(rendered_image)
            print("Final simulation frame saved.")

        final_poses = self._extract_body_poses(m, d)

        # Extract final poses of all bodies
        print(f"Extracted poses for {len(final_poses)} bodies")

        return final_poses


def item_from_hope(hope_item: Path, mapping=None) -> GroceryItem:
    """Convert a HOPE item to a GroceryItem."""

    if not hope_item.exists():
        raise FileNotFoundError(f"HOPE item path {hope_item} does not exist.")

    hope_item = hope_item / "google_16k"
    mesh_path = hope_item / "textured.obj"
    name = hope_item.parent.stem

    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
    mesh.compute_vertex_normals()

    # loaded_obb = mesh.get_oriented_bounding_box()
    loaded_obb = mesh.get_axis_aligned_bounding_box()
    extent = loaded_obb.max_bound - loaded_obb.min_bound

    try:
        # Construct convex hull from mesh vertices
        hull, _ = mesh.compute_convex_hull()

        # Check if convex hull is valid and watertight
        if not hull.is_watertight():
            raise ValueError(
                f"Convex hull is not watertight for mesh: {mesh_path}"
            )

        # Calculate volume from convex hull
        volume = hull.get_volume()

    except Exception as e:
        raise ValueError(
            f"Failed to calculate convex hull volume for mesh {mesh_path}: {e}"
        )
        
    if mapping is not None and name in mapping:
        friendly_name = mapping[name]["pretty_name"]
    else:
        logger.warning(f"No mapping found for item {name}, using raw name.")
        friendly_name = name

    # convert to m3
    volume = volume * 1e-6  # cm^3 to m^3

    sim_spec = SimSpec(
        name=name,
        mesh_path=mesh_path.as_posix(),
        import_transform=np.eye(4),  # final_import_transform
        volume=volume,
    )

    # convert from cm to m
    extent = extent / 100.0

    return GroceryItem(
        name=name,
        friendly_name=friendly_name,
        center=None,
        extent=extent,
        rotation=None,
        obb_target=None,
        packer_spec=None,
        sim_spec=sim_spec,
    )


if __name__ == "__main__":
    # Example usage
    sim = MujocoSim()

    # Define box size in meters (was [20, 30, 20] cm)
    box_size = np.array([0.20, 0.30, 0.20])  # (x, y, z) in meters
    sim.set_box(box_size, box_alpha=0.5)

    # Add grocery items from HOPE dataset
    hope_dataset_path = Path(
        "/home/blei/BagBuddy/dataset/hope-dataset/HOPE_3D_models"
    )  # Update with actual path
    item_names = [
        "BBQSauce",
        "Butter",
        "Corn",
        "GreenBeans",
        "Ketchup",
    ]  # Example item names
    for i, item_name in enumerate(item_names):
        hope_item_path = hope_dataset_path / item_name
        grocery_item = item_from_hope(hope_item_path)

        # Position items inside the box with origin at front-left corner
        x = 0.05 + (i % 2) * 0.1  # Distribute along width
        y = 0.05 + (i // 2) * 0.1  # Distribute along depth
        z = 0.1  # Height above bottom

        sample_obb = o3d.geometry.OrientedBoundingBox(
            center=np.array([x, y, z]), R=np.eye(3), extent=grocery_item.extent
        )
        grocery_item.obb_target = sample_obb

        sim.add_hope_grocery_item(grocery_item)

    # Run the simulation
    sim.run_sim(fps=30, speed=2.0, static=False)
