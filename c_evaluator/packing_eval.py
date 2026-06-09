from c_evaluator.sim import MujocoSim, item_from_hope
import pandas as pd
import json
import pickle
import base64
from c_evaluator.grocery_item import GroceryItem
import numpy as np
from pathlib import Path
from c_evaluator.viewer.save_viewer import SaveViewer
from c_evaluator.viewer.multi_viewer import MultiViewer
from c_evaluator.packing_eval_utils import (
    get_footprint,
    get_density,
    get_overlaps,
)
from dataclasses import dataclass
from tqdm import tqdm
from c_evaluator.import_poll_results import PollEvaluator
import time

# set np seed
seed=42
np.random.seed(seed)


def get_grocery_set(
    n_hope: int = 5, n_custom: int = 5, auxiliary_info_path: Path = None
) -> list[GroceryItem]:
    """
    Get a predefined set of grocery items for testing.

    Args:
        n_hope (int, optional): Number of HOPE items to include. Defaults to 5.
        n_custom (int, optional): Number of custom items to include.
            Defaults to 5.
    """

    # hope_path = Path("/home/blei/BagBuddy/dataset/hope-dataset/HOPE_3D_models")
    hope_path = Path(__file__).parent / "data" / "HOPE_3D_models"
    global seed
    np.random.seed(seed)
    seed += 1
    hope_set = get_grocery_set_hope(
        hope_path, n_items=n_hope, mapping_path=auxiliary_info_path
    )

    extension_path = Path(__file__).parent / "data" / "hope-extension"
    extension_set = get_grocery_set_hope(
        extension_path, n_items=n_custom, mapping_path=auxiliary_info_path
    )

    final_set = hope_set + extension_set

    return final_set


def pack_and_get_kpi(
    inference_function: callable,
    grocery_items: list[GroceryItem],
    boxes: dict[str, tuple[float, float, float]],
    visualizer: SaveViewer,
    enable_gui: bool = False,
    static: bool = False,
):
    start_time = time.time()
    items = inference_function(grocery_items, boxes, visualizer)
    packing_time = time.time() - start_time

    assert len(boxes) == 1, "Only one box size should be provided."
    box_size = list(boxes.values())[0]

    # add the box to be packed
    sim = MujocoSim(visualizer=visualizer)
    sim.set_box(box_size, box_alpha=0.5)

    for grocery_item in items:
        # fake rotation to eye
        # grocery_item.obb_target.R = np.eye(3)
        # add item to sim
        sim.add_hope_grocery_item(grocery_item)

    # Run the simulation for 5 seconds of sim time and get final poses
    final_poses = sim.run_sim(
        fps=30,
        speed=2.0,
        static=static,
        use_gui=enable_gui,
        settle_time=0.5,
        max_sim_time=5.0,
    )

    # Print final poses of all grocery items
    print("\nFinal poses of grocery items:")
    for item_name, pose in final_poses.items():
        if item_name != "cardboard_box":  # Skip the box itself
            pos = pose["position"]
            quat = pose["quaternion"]
            print(
                f"{item_name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                f"quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
            )

    for item in items:
        mj_name = sim.item_to_name(item)
        if mj_name not in final_poses:
            raise ValueError(
                f"Item {mj_name} not found in final poses from " "simulation."
            )

        final_pose = final_poses[mj_name]
        footprint = get_footprint(item, final_pose, box_size, grid_resolution=0.005)
        assert item.sim_spec is not None, "SimSpec must be initialized."
        item.sim_spec.final_pose = final_pose
        item.sim_spec.footprint = footprint

        visualizer.add_footprint(footprint, item.name)

    # compute overlaps
    overlaps = get_overlaps(items)
    visualizer.add_overlap_dataframe(overlaps)
    density = get_density(items, box_size)

    return density, overlaps, packing_time


def get_grocery_set_hope(
    path: Path, n_items: int = 5, mapping_path: Path = None
) -> list[GroceryItem]:
    """Get a predefined number of grocery items from the HOPE dataset.

    Returns:
        _type_: _description_
    """
    # open mapping json file if provided
    if mapping_path is not None:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        print(f"Loaded mapping for {len(mapping)} items from {mapping_path}.")
    # Setup and load dataset
    item_folders = [f for f in path.iterdir() if f.is_dir()]
    item_folders = sorted(item_folders, key=lambda p: p.name)
    grocery_items = []
    # for item_folder in item_folders:
    #    grocery_items.append(item_from_hope(item_folder))

    # add 5 random samples to bagbuddy
    for grocery_item_path in np.random.choice(
        item_folders, size=n_items, replace=False
    ):
        # print the names
        grocery_item = item_from_hope(grocery_item_path, mapping=mapping)
        print(f"Adding item: {grocery_item.name} with extent {grocery_item.extent} m")
        grocery_items.append(grocery_item)

    return grocery_items


def get_box_proposals_interpolate(start_box, end_box, steps, min_width):
    box_list = []
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_box = (
            max(start_box[2] * (1 - alpha) + end_box[2] * alpha, min_width),
            start_box[0] * (1 - alpha) + end_box[0] * alpha,
            start_box[1] * (1 - alpha) + end_box[1] * alpha,
        )
        interpolated_box = np.array(interpolated_box)
        interpolated_box.flags.writeable = False
        box_list.append(interpolated_box)

    box_list = [{f"{int(box[0])}x{int(box[1])}x{int(box[2])}": box} for box in box_list]

    return box_list


@dataclass
class ExperimentResult:
    box_size: tuple
    density: list
    grocery_items: list[GroceryItem]
    overlaps: pd.DataFrame
    c_score: float
    n_violations: int
    n_total: int
    packing_time: float

    def to_dict(self):
        # Pickle and base64 encode the overlaps DataFrame to handle duplicates
        pickled_overlaps = pickle.dumps(self.overlaps)
        encoded_overlaps = base64.b64encode(pickled_overlaps).decode("utf-8")

        return {
            "box_size": str(self.box_size),
            "density": self.density.tolist(),
            "grocery_items": [item.name for item in self.grocery_items],
            "overlaps": encoded_overlaps,
            "c_score": self.c_score,
            "n_violations": int(self.n_violations),
            "n_total": int(self.n_total),
            "packing_time": self.packing_time,
        }


def run_exp_once(
    inferenece_function: callable,
    n_hope: int = 5,
    n_custom: int = 4,
    exp_id: int = 0,
    enable_gui: bool = False,
    static: bool = False,
    hope_mapping_path: Path = None,
    output_dir: Path = Path("./experiment_output"),
    collision_padding = 0.0
):

    hope_mapping_path = Path(__file__).parent / "data" / "hope_mapping.json"

    # generate grocery set
    grocery_items = get_grocery_set(
        n_hope=n_hope,
        n_custom=n_custom,
        auxiliary_info_path=hope_mapping_path,
    )

    max_size = np.zeros(3)
    for item in grocery_items:
        max_size = np.maximum(max_size, item.extent)

    minimal_box_length = np.max(
        max_size
    )  # one side of the box must be at least as large as the largest item dimension
    minimal_box_length = minimal_box_length + 0.005 + 2 * collision_padding  # add 1cm buffer

    # define box sizes to evaluate in cm
    #boxes = get_box_proposals_interpolate(
    #    start_box=(0.40, 0.40, 0.40),
    #    end_box=(0.10, 0.10, 0.10),
    #    steps=10,
    #    min_width=minimal_box_length,
    #)

    # main save visualizer to .visualize
    view_visualizer = SaveViewer(output_dir=output_dir)

    results = []
    terminated = False
    box_initial_guess = np.array([0.5, 0.5, 0.5], dtype=float)  # start from a large box

    box = box_initial_guess.copy()
    box[0] = max(box[0], minimal_box_length)

    best_success_box = None
    worst_failure_box = None
    previous_density = None
    iteration = 0
    max_iterations = 100
    box_convergence_tol = 2e-3  # 2 mm tolerance on each side
    shrink_factor = 0.5
    growth_factor = 2.0

    while True:
        iteration += 1
        if iteration > max_iterations:
            if best_success_box is not None:
                print(
                    "Binary search hit iteration limit; returning best successful box size "
                    f"{best_success_box}."
                )
                terminated = True
                break
            raise RuntimeError("Binary search did not converge within the maximum number of iterations.")
        print(f"Evaluating box size: {box}")

        save_visualizer = SaveViewer(output_dir=output_dir)
        visualizer = MultiViewer([view_visualizer, save_visualizer])

        output_dir.mkdir(parents=True, exist_ok=True)

        # run experiment
        box_dict = {f"{int(box[0])}x{int(box[1])}x{int(box[2])}": box}
        try:
            density, overlaps, packing_time = pack_and_get_kpi(
                inferenece_function,
                grocery_items,
                enable_gui=enable_gui,
                boxes=box_dict,
                visualizer=visualizer,
                static=static,
            )

            # calculate c score
            response_path = Path(__file__).parent / "data" / "responses.csv"
            poll_evaluator = PollEvaluator(response_path, hope_mapping_path)
            c_score, n_failed, n_violations, n_total = (
                poll_evaluator.get_rel_score_mult_mat(overlaps)
            )
            print(f"C-score for box size {list(box_dict.keys())[0]}: {c_score:.2f}")

            assert (
                len(n_violations) == 1 and len(n_total) == 1
            ), "Expected single value for violations and total."
            results.append(
                ExperimentResult(
                    box_size=box,
                    density=density,
                    grocery_items=grocery_items,
                    overlaps=overlaps,
                    c_score=c_score,
                    n_violations=n_violations[0],
                    n_total=n_total[0],
                    packing_time=packing_time,
                )
            )
            success = True
        except RuntimeError as e:
            #print(f"Packing failed for box {box}.")
            #final_density = results[-1].density if results else "unknown"
            #print(f"Last density is {final_density*100:.2f}%, finished Evaluation.")
            #terminated = True
            # output_dir.rmdir()
            #break
            success = False
            density = None
        
        # binary search logic 
        if success:
            best_success_box = box.copy()
            if previous_density is not None and abs(previous_density - density) < 0.01:
                print(
                    f"Terminating experiment for exp_id {exp_id} at box size {best_success_box}."
                )
                terminated = True
                break
            previous_density = density
        else:
            worst_failure_box = box.copy()

        if best_success_box is not None and worst_failure_box is not None:
            bracket_span = np.abs(best_success_box - worst_failure_box)
            if np.all(bracket_span <= box_convergence_tol):
                print(
                    "Box bracket converged within tolerance; terminating search at box size "
                    f"{best_success_box}."
                )
                terminated = True
                break
            next_box = worst_failure_box + (best_success_box - worst_failure_box) / 2.0
        elif best_success_box is not None:
            next_box = best_success_box * shrink_factor
        elif worst_failure_box is not None:
            next_box = worst_failure_box * growth_factor
        else:
            raise RuntimeError("Binary search state invalid: no success or failure recorded.")

        next_box = next_box.astype(float)
        next_box[0] = max(next_box[0], minimal_box_length)  # enforce minimum along primary axis
        next_box = np.maximum(next_box, 1e-3)  # keep strictly positive dimensions

        if np.allclose(next_box, box, atol=1e-4):
            if success and best_success_box is not None:
                print(
                    "Box update stalled after a successful run; using current box size as optimum."
                )
                terminated = True
                break
            print("Box update stalled after a failure; expanding search region.")
            if worst_failure_box is None:
                box = box * growth_factor
            else:
                box = worst_failure_box * growth_factor
            box = box.astype(float)
            box[0] = max(box[0], minimal_box_length)
            continue

        box = next_box

        # dump all the experiment results to a json file
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=4)

    if not terminated:
        raise ValueError("Experiment did not terminate early as expected.")


def run_exp(
    inference_function: callable,
    exp_name: str,
    enable_gui: bool = False,
    output_dir: Path = Path("./experiment_output"),
    static: bool = False,
    n_hope: int = 5,
    n_custom: int = 4,
    collision_padding=0.0 # important for selecting the right box size
):
    n_experiments = 10

    # make sure the output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_id in range(n_experiments):
        run_exp_once(
            inference_function,
            n_hope=n_hope,
            n_custom=n_custom,
            exp_id=exp_name + f"_{exp_id}",
            enable_gui=enable_gui,
            output_dir=output_dir / f"{exp_name}/{exp_id}",
            static=static,
            collision_padding = collision_padding
        )


def run_scaling_exp(
    inference_function: callable,
    exp_name,
    enable_gui: bool = False,
    output_dir: Path = Path("./experiment_output"),
    static: bool = False,
    start_stop_step: tuple = (2, 10, 2),
    fragile_percentage: float = 0.3
):

    # make sure the output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_items in reversed(range(
        start_stop_step[0], start_stop_step[1] + 1, start_stop_step[2]
    )):
        n_hope = int(n_items * (1 - fragile_percentage))
        n_custom = n_items - n_hope
        run_exp(
            inference_function,
            exp_name=exp_name,
            n_hope=n_hope,
            n_custom=n_custom,
            enable_gui=enable_gui,
            output_dir=output_dir / f"{exp_name}/n{n_items}",
            static=static,
        )


if __name__ == "__main__":
    run_exp()
