from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import List
from ipack_evaluator.grocery_item import GroceryItem


logger = logging.getLogger(__name__)


class ViewerBase(ABC):
    @abstractmethod
    def visualize_packing(
        self,
        box_size: tuple,
        items: list,
        title: str = None,
        savepath: str = None,
        show: bool = True,
        alpha: float = 0.5,
    ):
        pass

    @abstractmethod
    def visualize_groceries(self, items: List[GroceryItem]):
        """Visualize grocery items."""
        pass

    @abstractmethod
    def visualize_sensors(self, image: np.ndarray):
        """Visualize sensor data (e.g., camera image)."""
        pass

    @abstractmethod
    def add_footprint(self, grid: np.ndarray, name: str):
        """
        Visualize a 2D occupancy grid as a set of xy plane.
        The np.ndarray should be a 2D binary grid where 1 indicates
        occupied cells.

        Args:
            grid (np.ndarray): 2D binary occupancy grid
            name (str): Name for the footprint visualization
        """
        pass

    @abstractmethod
    def add_packing_video(
        self,
        frames: List[np.ndarray],
        fps: int = 30,
    ):
        """Create a video visualization of the packing process.
        
        Args:
            frames: List of image frames (as numpy arrays)
            fps: Frames per second for the video
        """
        pass
