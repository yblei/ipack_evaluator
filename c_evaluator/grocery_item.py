# replica of the llm_pack grocery item class
import open3d as o3d
import numpy as np
import uuid
from typing import Union, TypeVar
from dataclasses import dataclass

@dataclass
class SimSpec:
    name: str
    mesh_path: str
    import_transform: np.ndarray  # shape (4, 4)
    volume: float  # in cubic meters
    final_pose: dict = None  # {'position': [x,y,z], 'quaternion': [w,x,y,z]}
    footprint: np.ndarray = None  # 2D binary occupancy grid

@dataclass
class PackerSpec:
    shortest_z_up: bool
    allowed_z: list[str]

@dataclass()
class GroceryItem:
    name: str
    friendly_name: str = None
    uuid: str = str(uuid.uuid4())
    center: Union[np.ndarray, None] = None  # shape (3,)
    extent: Union[np.ndarray, None] = None  # shape (3,)
    rotation: Union[np.ndarray, None] = None  # shape (3, 3)
    obb_target: Union[o3d.geometry.OrientedBoundingBox, None] = None
    packer_spec: Union[PackerSpec, None] = None
    sim_spec: Union[SimSpec, None] = None

    @property
    def obb_is(self) -> o3d.geometry.OrientedBoundingBox:
        """Get the oriented bounding box for this grocery item."""
        obb = o3d.geometry.OrientedBoundingBox(
            self.center, self.rotation, self.extent
        )
        return obb
    
    def __hash__(self) -> int:
        return hash(f"{self.name}_{self.uuid}")
    
    def __str__(self) -> str:
        return f"GroceryItem(name={self.name})"
    
    def to_dict(self) -> dict:
        """Convert the GroceryItem to a dictionary."""
        return {
            "name": self.name,
            "uuid": self.uuid,
            "center": self.center.tolist() if self.center is not None else None,
            "extent": self.extent.tolist() if self.extent is not None else None,
            "rotation": self.rotation.tolist() if self.rotation is not None else None,
            "obb_target": {
                "center": self.obb_target.center.tolist(),
                "extent": self.obb_target.extent.tolist(),
                "rotation": self.obb_target.R.tolist(),
            } if self.obb_target is not None else None,
            "packer_spec": {
                "shortest_z_up": self.packer_spec.shortest_z_up,
                "allowed_z": self.packer_spec.allowed_z,
            } if self.packer_spec is not None else None,
        }

    @staticmethod
    def from_obb(
        name: str, obb: o3d.geometry.OrientedBoundingBox
    ) -> "GroceryItem":
        """Create a GroceryItem from an Open3D OrientedBoundingBox.

        Args:
            name (str): The name of the grocery item.
            obb (o3d.geometry.OrientedBoundingBox): The oriented bounding box.

        Returns:
            GroceryItem: The created grocery item.
        """
        center = obb.center
        extent = obb.extent
        rotation = obb.R

        return GroceryItem(name, center, extent, rotation)
    
if __name__ == "__main__":
    # test grocery item creation
    obb = o3d.geometry.OrientedBoundingBox(
        center=np.array([0.0, 0.0, 0.0]),
        R=np.eye(3),
        extent=np.array([0.1, 0.2, 0.3]),
    )
    item = GroceryItem.from_obb("test_item", obb)
    print(item)
    print(item.to_dict())