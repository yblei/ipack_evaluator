import numpy as np
from pathlib import Path

OBJ_PATH = Path(__file__).parent / "textured.obj"

def get_obj_bounding_box(obj_path):
	vmin = np.array([np.inf, np.inf, np.inf])
	vmax = np.array([-np.inf, -np.inf, -np.inf])
	with open(obj_path, 'r') as f:
		for line in f:
			if line.startswith('v '):
				parts = line.strip().split()
				if len(parts) == 4:
					v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
					vmin = np.minimum(vmin, v)
					vmax = np.maximum(vmax, v)
	return vmin, vmax

def main():
	vmin, vmax = get_obj_bounding_box(OBJ_PATH)
	print(f"Bounding box min: {vmin}")
	print(f"Bounding box max: {vmax}")
	print(f"Size (x, y, z): {vmax - vmin}")

if __name__ == "__main__":
	main()
