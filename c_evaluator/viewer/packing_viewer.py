
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl
mpl.use("Agg")  # Use a non-interactive backend

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe  # NEW: for text halo
import numpy as np
from python_tex_tools import TexExporter, make_plt_look_like_latex

def _cuboid_faces(origin, size):
    ox, oy, oz = origin
    lx, ly, lz = size
    x0, x1 = ox, ox + lx
    y0, y1 = oy, oy + ly
    z0, z1 = oz, oz + lz
    # 8 vertices
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    # 6 faces defined by the 4-vertex loops
    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[0], v[3], v[7], v[4]],  # left
    ]

def plot_packing(box_size, items, title=None, savepath=None, show=True, alpha=0.5, enable_text=True):
    """
    Visualize a 3D bin packing solution.

    Args:
        box_size: (W, D, H) tuple for the chosen box.
        items: list of items to visualize, each with:
            {
              "name": str,
              "pos": (x, y, z),    # lower-left-near corner (axis-aligned)
              "size": (w, d, h)    # oriented size
            }
        title: optional string figure title
        savepath: optional path to save image (e.g., "packing.png")
        show: whether to call plt.show()
        alpha: face alpha for cuboids
    """

    enable_text = False 
    
    placements = []
    for item in items:
        item_converted ={
        }
        
        pos=item.obb_target.center
        size = item.obb_target.extent
        name = item.name
        
        # apply the rotation
        item_converted["size"] = np.abs(item.obb_target.R @ size)
        item_converted["pos"] = pos - item_converted["size"] / 2.0
        item_converted["name"] = name
        
        placements.append(item_converted)
    
    # convert objects to cm
    for item in placements:
        item["pos"] = [x * 100 for x in item["pos"]]
        item["size"] = [x * 100 for x in item["size"]]

    W, D, H = box_size
    # convert to cm
    W *= 100
    D *= 100
    H *= 100

    with make_plt_look_like_latex():
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Draw outer box as a wireframe (thin)
        faces = _cuboid_faces((0, 0, 0), (W, D, H))
        outer = Poly3DCollection(faces, linewidths=1, facecolors=None, alpha=0.0)
        ax.add_collection3d(outer)

        # Draw each item
        for it in placements:
            px, py, pz = it["pos"]
            sx, sy, sz = it["size"]
            faces = _cuboid_faces((px, py, pz), (sx, sy, sz))
            poly = Poly3DCollection(faces, linewidths=1.5, edgecolors='black', alpha=alpha)
            ax.add_collection3d(poly)
            # Label slightly above the item (improved readability)
            lx = px + sx * 0.5
            ly = py + sy * 0.5
            lz = pz + sz + max(0.5, 0.02*H)
            if enable_text:
                txt = ax.text(
                    lx, ly, lz, it["name"],
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
                )
                # Add a white halo around black text
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

        ax.set_xlim(0, W)
        ax.set_ylim(0, D)
        ax.set_zlim(0, H)
        ax.set_box_aspect((W, D, H))  # equal aspect

        ax.set_xlabel("x (width, cm)")
        ax.set_ylabel("y (depth, cm)")
        ax.set_zlabel("z (height, cm)")
        #if title:
        #    ax.set_title(title)
        pdf_path=str(savepath).replace("png","pdf") if savepath else None
        if savepath:
            plt.savefig(savepath, dpi=160)
            plt.savefig(pdf_path, dpi=160, bbox_inches='tight', pad_inches=0.25)
        if show:
            plt.show()

    # export latex

def placements_from_pyscipopt(model, items, rotations_dict, x, y, z, r):
    """
    Extract placements from a PySCIPOpt model that uses variables:
      - x[i], y[i], z[i]: continuous placement variables
      - r[(i,p)]: binary rotation selector for p in {0..5}
    rotations_dict: P[i] -> list of 6 oriented size tuples (w,d,h).
    Returns (box_size, placements).
    Assumes that the chosen box extents are the max of coordinates + sizes.
    If you used explicit box variables, adapt as needed to read them instead.
    """
    # Infer box extents from items -- adapt if you model box choice explicitly
    max_x = 0.0; max_y = 0.0; max_z = 0.0
    placements = []
    for i in items:
        # chosen rotation
        chosen = 0
        for p in range(6):
            if model.getVal(r[(i, p)]) > 0.5:
                chosen = p
                break
        w, d, h = rotations_dict[i][chosen]
        xi = model.getVal(x[i]); yi = model.getVal(y[i]); zi = model.getVal(z[i])
        placements.append({"name": i, "pos": (xi, yi, zi), "size": (w, d, h)})
        max_x = max(max_x, xi + w)
        max_y = max(max_y, yi + d)
        max_z = max(max_z, zi + h)

    box_size = (max_x, max_y, max_z)
    return box_size, placements

if __name__ == "__main__":
    # Minimal demo with fake placements
    groceries = {
        "apple":  ((10, 10, 10), (10, 10, 10), (10, 10, 10), (10, 10, 10), (10, 10, 10), (10, 10, 10)),
        "banana": ((15, 5, 5),   (15, 5, 5),   (5, 15, 5),   (5, 15, 5),   (5, 5, 15),   (5, 5, 15)),
        "orange": ((8, 8, 8),    (8, 8, 8),    (8, 8, 8),    (8, 8, 8),    (8, 8, 8),    (8, 8, 8)),
        "milk":   ((20, 10, 10), (20, 10, 10), (10, 20, 10), (10, 20, 10), (10, 10, 20), (10, 10, 20)),
        "bread":  ((15, 10, 10), (15, 10, 10), (10, 15, 10), (10, 15, 10), (10, 10, 15), (10, 10, 15)),
    }
    box_size = (40, 20, 20)
    placements = [
        {"name": "apple",  "pos": (0, 0, 0),  "size": (10, 10, 10)},
        {"name": "banana", "pos": (10, 0, 0), "size": (15, 5, 5)},
        {"name": "orange", "pos": (25, 0, 0), "size": (8, 8, 8)},
        {"name": "milk",   "pos": (0, 10, 0), "size": (20, 10, 10)},
        {"name": "bread",  "pos": (20, 10, 0),"size": (15, 10, 10)},
    ]
    plot_packing(box_size, placements, title="Packing demo", savepath="packing_demo.png", show=False)