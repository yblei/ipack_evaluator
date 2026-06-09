from ipack_evaluator.viewer.viewer_base import ViewerBase
from ipack_evaluator.viewer.packing_viewer import plot_packing
from ipack_evaluator.grocery_item import GroceryItem
from typing import List
import numpy as np

class SaveViewer(ViewerBase):
    """A simple viewer that just saves everything to disk.  No GUI."""
    def __init__(self, output_dir) -> None:
        super().__init__()
        self.output_dir = output_dir

    def visualize_packing(
        self,
        box_size: tuple,
        items: list,
        title: str = None,
        alpha: float = 0.5,
    ):
        plot_packing(
            box_size,
            items,
            title=title,
            savepath=self.output_dir / "packing.png",
            show=False,
            alpha=alpha,
        )
        
    def visualize_groceries(self, items: list[GroceryItem]):
        """Save images of each grocery item."""
        
        # write to json
        import json
        from pathlib import Path
        items_json = [item.to_dict() for item in items]
               
        items_overview = []
        for item in items:
            name = item.name
            pretty_extent = ", ".join([f"{e*100:.2f}cm" for e in item.extent])
            items_overview.append({"name": name, "extent": pretty_extent})
            
        
        with open(self.output_dir / "perception_overview.json", "w") as f:
            json.dump(items_overview, f, indent=4)

        with open(self.output_dir / "perception_results.json", "w") as f:
            json.dump(items_json, f, indent=4)
            
    def visualize_sensors(self, image: np.ndarray):
        """Save the perception image."""
        import cv2
        cv2.imwrite(str(self.output_dir / "perception_image.png"), image)
        
    def save_sim_box_render(self, image: np.ndarray):
        """Save the simulation box render image."""
        import cv2
        cv2.imwrite(str(self.output_dir / "sim_box_render.png"), image)
        
    def add_footprint(self, grid: np.ndarray, name: str):
        """
        Plotts a 2d occupancy grid as a set of xy plane.
        The np.ndarray should be a 2D binary grid where 1 indicates occupied cells.
        saves the resulting plot to disk.

        Args:
            grid (np.ndarray): _description_
            name (str): _description_
        """
        
        # use matplotlib, dont show
        import matplotlib.pyplot as plt
        # reset mpl
        plt.clf()
        
        plt.imshow(grid, cmap='gray', origin='lower')
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Occupancy')
        plt.grid(False)
        
        file_path = self.output_dir / f"footprint_{name}.png"

        plt.savefig(file_path)
        plt.close()
         
        
    def add_packing_video(
        self, 
        frames: List[np.ndarray], fps: int = 30,
    ):
        """Save a video of the packing process."""
        import cv2
        height, width, _ = list(np.ndarray)[0].shape
        video_path = self.output_dir / "packing_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in list(np.ndarray):
            video_writer.write(frame)
        
        video_writer.release()

    def visualize_no_under_pairs(self, pairs):

        out = []
        
        for pair in pairs: 
            top = pair[0].name
            bottom = pair[1].name
            out.append((top, bottom))

        import json 
        with open(self.output_dir / "no_under_pairs.json", "w") as f:
            json.dump(out, f, indent=4)
        
    def add_overlap_dataframe(self, df):
        """Save the overlap dataframe to a heatmap image."""
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.clf()
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Overlap Heatmap")
        plt.xlabel("Item")
        plt.ylabel("Item")
        
        file_path = self.output_dir / "overlap_heatmap.png"
        plt.savefig(file_path)
        plt.close()
