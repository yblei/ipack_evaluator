from ipack_evaluator.packing_eval import run_exp
import open3d as o3d
import numpy as np

def test_inference(grocery_items, boxes, visualizer):
    
    # grid x y placement of items in the box
    n_items = len(grocery_items)

    d_z = 0.1

    for i, item in enumerate(grocery_items):
        # example values
        center = np.array([0.3,0.3,0.1 * i])
        extent = item.extent
        rotation = np.eye(3)
        
        obb_target = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        item.obb_target = obb_target
        
    return grocery_items
    

if __name__ == "__main__":
    
    run_exp(inference_function=test_inference, exp_name="test_inference", enable_gui=True)
    