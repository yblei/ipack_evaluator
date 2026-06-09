import argparse
import sys
from functools import partial

from bagbuddy.bagbuddy.main import BagBuddy
from bagbuddy.bagbuddy.utils import PackerSpec
from bagbuddy.fm_wrapper import GPTWrapper, ClaudeWrapper

sys.path.append("/home/blei/BagBuddy/evaluation/c_evaluator")
from c_evaluator.packing_eval import run_exp, run_scaling_exp


def run_bb_evaluation(
    grocery_items, boxes, visualizer, use_vanilla_llm_pack, vlm: str, output_dir: str, simulate_hw_constraints = False, collision_padding = 0.00
):
    """
    Takes the grocery items and runs the BagBuddy planner to plan the packing order.
    returns the grocery items with obb_target set.
    """
    use_cached_sensor_data = False

    detector = None

    boxes = {k: {"size": v} for k, v in boxes.items()}

    system_prompt = (
        """You are an intelligent AI, assisting a robot in a grocery store."""
    )

    if use_cached_sensor_data:
        if "gpt" in vlm:
            vlm = GPTWrapper(
                model=vlm,
                enable_caching=True,
                simulate_time_delay=True,
                system_prompt=system_prompt,
            )
        elif "claude" in vlm:
            vlm = ClaudeWrapper(
                model=vlm,
                enable_caching=True,
                simulate_time_delay=True,
                system_prompt=system_prompt,
            )
        elif "gemini" in vlm:
            from bagbuddy.fm_wrapper import GeminiWrapper

            vlm = GeminiWrapper(
                model=vlm,
                enable_caching=True,
                simulate_time_delay=True,
                system_prompt=system_prompt,
            )
    else:
        vlm = GPTWrapper(system_prompt=system_prompt)

    bb = BagBuddy(
        vlm,
        detector,
        use_caching=use_cached_sensor_data,
        viewer=visualizer,
        boxes=boxes,
        use_robot=False,
        disable_robot=True,
    )

    if simulate_hw_constraints:
        for item in grocery_items:
            spec = PackerSpec(shortest_z_up=True, allowed_z=["h", "w", "d"])
            item.packer_spec = spec
            bb.planner.enforce_base_level_constraint = True

    bb.planner.use_vanilla_llm_pack = use_vanilla_llm_pack
    #bb.planner.boxes = boxes
    bb.grocery_items = grocery_items

    # run the planner: this adds the obb_target to each grocery item
    bb.plan_order(collision_padding)

    items = bb.grocery_items
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BagBuddy LLM packing strategies")
    parser.add_argument(
        "--vlm",
        default="gpt-5-mini",
        help="Vision-language model identifier (e.g., gpt-5-mini, gpt-5, claude-sonnet-4-20250514, gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--experiment-type",
        choices=["basic", "hw", "scaling"],
        default="hw",
        help="Evaluation mode: basic scenario, hardware constraints simulation, or scaling study",
    )
    parser.add_argument(
        "--vanilla-llm-pack",
        action="store_true",
        help="Use the baseline vanilla LLM packing planner instead of optimized planner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_vanilla_llm_pack = args.vanilla_llm_pack
    vlm = args.vlm
    experiment_type = args.experiment_type

    if use_vanilla_llm_pack:
        exp_name = "vanilla_llm_pack"
    else:
        exp_name = f"llm_pack_{vlm}"

    collision_padding = 0.0
    sim_hw = False
    if experiment_type == "hw":
        sim_hw = True

    inference_function = partial(
        run_bb_evaluation,
        vlm=vlm,
        use_vanilla_llm_pack=use_vanilla_llm_pack,
        output_dir=f"experiment_output/{exp_name}",
        simulate_hw_constraints=sim_hw,
        collision_padding=collision_padding,
    )

    if experiment_type == "basic":
        run_exp(inference_function, exp_name=exp_name, enable_gui=False, static=True)
    elif experiment_type == "hw":
        exp_name = f"{exp_name}_hw"
        run_exp(
            inference_function,
            exp_name=exp_name,
            enable_gui=False,
            static=True,
            collision_padding=0.01,
        )
    elif experiment_type == "scaling":
        exp_name = f"{exp_name}_scaling"
        run_scaling_exp(
            inference_function,
            exp_name=exp_name,
            enable_gui=False,
            static=True,
            start_stop_step=(18, 20, 2),
        )
    else:
        raise ValueError(f"Invalid experiment type {experiment_type}")


if __name__ == "__main__":
    main()
