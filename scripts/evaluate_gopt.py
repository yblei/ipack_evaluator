import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure GOPT code and local scripts are importable
sys.path.insert(0, '/home/blei/BagBuddy/evaluation/c_evaluator/GOPT')

# Import c_evaluator utilities
from c_evaluator.packing_eval import run_exp
import open3d as o3d

import gymnasium as gym
import torch
from omegaconf import OmegaConf
from tianshou.data import Batch
from tianshou.utils.net.common import ActorCritic

from masked_ppo import MaskedPPOPolicy
from tools import CategoricalMasked, registration_envs, set_seed
from ts_train import build_net
from scipy.spatial.transform import Rotation as R


def _abs_config_path(config_path: str) -> str:
    if os.path.isabs(config_path):
        return config_path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, config_path)


def load_cfg(
    config_path: str,
    *,
    ckp: Optional[str] = None,
    cuda: bool = True,
    device: int = 0,
    seed: Optional[int] = None,
    render: bool = False,
) -> Any:
    seed=None
    config_path = _abs_config_path(config_path)
    cfg = OmegaConf.load(config_path)

    box_small = int(max(cfg.env.container_size) / 10)
    box_big = int(max(cfg.env.container_size) / 2)
    box_range = (box_small, box_small, box_small, box_big, box_big, box_big)

    if cfg.get("env.step") is not None:
        step = cfg.env.step
    else:
        step = box_small

    box_size_set: List[Tuple[int, int, int]] = []
    for i in range(box_range[0], box_range[3] + 1, step):
        for j in range(box_range[1], box_range[4] + 1, step):
            for k in range(box_range[2], box_range[5] + 1, step):
                box_size_set.append((i, j, k))

    cfg.env.box_small = box_small
    cfg.env.box_big = box_big
    cfg.env.box_size_set = box_size_set

    cfg.config = config_path
    cfg.ckp = ckp
    cfg.cuda = bool(cuda)
    cfg.device = int(device)
    cfg.render = bool(render)

    if seed is not None:
        cfg.seed = int(seed)

    if hasattr(cfg, "train") and hasattr(cfg.train, "algo"):
        cfg.train.algo = str(cfg.train.algo).upper()

    return cfg


@dataclass
class PackedBox:
    size_xyz: Tuple[int, int, int]
    pos_xyz: Tuple[int, int, int]
    rotated: bool


@dataclass
class InferenceResult:
    ratio: float
    counter: int
    boxes: List[PackedBox]
    info: Dict[str, Any]


class BoxPacker:
    def __init__(
        self,
        *,
        box_size_set = None,
        container_size = None,
        config: str = "../GOPT/cfg/config.yaml",
        checkpoint: str,
        device: Optional[str] = None,
    ) -> None:
        if box_size_set is None or container_size is None:
            raise NotImplementedError("Need to specify box size set and container size.")
        self.box_size_set = box_size_set
        self.container_size = container_size
        
        registration_envs()

        self.args = load_cfg(config, ckp=checkpoint, cuda=True, device=0)

        if device is not None:
            if device == "cpu":
                self.torch_device = torch.device("cpu")
            else:
                self.torch_device = torch.device(device)
        else:
            if self.args.cuda and torch.cuda.is_available():
                self.torch_device = torch.device("cuda", int(self.args.device))
            else:
                self.torch_device = torch.device("cpu")

        set_seed(self.args.seed, self.args.cuda, getattr(self.args, "cuda_deterministic", False))

        self.env = self._make_env(render=getattr(self.args, "render", False))
        self.policy = self._load_policy(self.env.action_space)

    def _make_env(self, *, render: bool = False):
        container_size_internal=[10,10,10]

        self.scale_factor = container_size_internal / self.container_size
        box_size_set = self.box_size_set

        groceries_scaled = np.int16(np.ceil(box_size_set * self.scale_factor))
        self.groceries_scaled = groceries_scaled


        return gym.make(
            self.args.env.id,
            container_size=container_size_internal,
            enable_rotation=self.args.env.rot,
            data_type="sequence",
            item_set=groceries_scaled,
            reward_type=self.args.train.reward_type,
            action_scheme=self.args.env.scheme,
            #k_placement=self.args.env.k_placement,
            k_placement=len(groceries_scaled),
            is_render=render,
        )

    def _load_policy(self, action_space) -> MaskedPPOPolicy:
        actor, critic = build_net(self.args, self.torch_device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=self.args.opt.lr, eps=self.args.opt.eps)

        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=CategoricalMasked,
            discount_factor=self.args.train.gamma,
            eps_clip=self.args.train.clip_param,
            advantage_normalization=False,
            vf_coef=self.args.loss.value,
            ent_coef=self.args.loss.entropy,
            gae_lambda=self.args.train.gae_lambda,
            action_space=action_space,
        )

        raw = torch.load(self.args.ckp, map_location=self.torch_device)
        state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        policy.load_state_dict(state)
        policy.eval()
        return policy

    @torch.no_grad()
    def pack_one_episode(self, *, render: Optional[bool] = None) -> InferenceResult:

        if render is not None and render != getattr(self.args, "render", False):
            try:
                self.env.close()
            except Exception:
                pass
            self.env = self._make_env(render=render)

        obs, info = self.env.reset()

        terminated = False
        truncated = False
        state = None
        last_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}

        while not (terminated or truncated):
            obs_batched = Batch(
                obs=np.expand_dims(obs["obs"], axis=0),
                mask=np.expand_dims(obs["mask"], axis=0),
            )
            batch = Batch(obs=obs_batched)
            out = self.policy(batch, state=state)
            state = out.state

            act = out.act
            action = int(act.item()) if hasattr(act, "item") else int(act[0])

            obs, rew, terminated, truncated, step_info = self.env.step(action)
            if isinstance(step_info, dict):
                last_info.update(step_info)

        boxes: List[PackedBox] = []
        try:
            packed_boxes = list(getattr(self.env, "container").boxes)
            rot_flags = list(getattr(self.env, "container").rot_flags)
            for b, rot in zip(packed_boxes, rot_flags):
                boxes.append(
                    PackedBox(
                        size_xyz=(int(b.size_x), int(b.size_y), int(b.size_z)),
                        pos_xyz=(int(b.pos_x), int(b.pos_y), int(b.pos_z)),
                        rotated=bool(rot),
                    )
                )
        except Exception:
            boxes = []

        ratio = float(last_info.get("ratio", 0.0))
        counter = int(last_info.get("counter", len(boxes)))

        return InferenceResult(ratio=ratio, counter=counter, boxes=boxes, info=last_info)


def _compute_axis_scales(box_dims_m, container_size_units):
    """Compute per-axis meter-per-unit scales from container size and desired box dims.

    box_dims_m: (w, h, d) in meters
    container_size_units: (W, H, D) in integer units from GOPT config
    returns: np.array([sx, sy, sz]) in meters per unit
    """
    box_dims_m = np.array(box_dims_m, dtype=float)
    container_size_units = np.array(container_size_units, dtype=float)
    # Avoid division by zero
    container_size_units = np.where(container_size_units == 0, 1.0, container_size_units)
    return box_dims_m / container_size_units


def _retrieve_rotation_matrix(placed_size: Tuple[int, int, int],
                              expected_size: Tuple[int, int, int]) -> np.ndarray:
    """Return a 3x3 rotation matrix that permutes axes so expected_size -> placed_size.

    We assume axis-aligned 90° rotations (permutations of axes). If multiple
    permutations fit, the first match is used. Raises if no permutation matches.
    """
    placed = tuple(int(x) for x in placed_size)
    expected = tuple(int(x) for x in expected_size)

    # Quick sanity: multisets must match to be a pure axis permutation
    if sorted(placed) != sorted(expected):
        raise ValueError(f"Sizes are not a pure axis permutation: expected={expected}, placed={placed}")
    rotation_matrices = []
    rotation_matrices.append(np.eye(3))  # Identity
    rotation_matrices.append(R.from_euler('z', 90, degrees=True).as_matrix())
    rotation_matrices.append(R.from_euler('y', 90, degrees=True).as_matrix())
    rotation_matrices.append(R.from_euler('x', 90, degrees=True).as_matrix())

    for R_mat in rotation_matrices:
        # Apply rotation to expected size
        rotated = R_mat @ np.array(expected)

        if np.allclose(abs(rotated), placed):
            return np.round(R_mat)
            #return np.eye(3)  # Placeholder for correct rotation matrix

    # Fallback should never trigger if multisets matched
    raise ValueError(f"No axis permutation found for expected={expected}, placed={placed}")


def run_gopt_evaluation(grocery_items, boxes, visualizer, render: bool = False):
    """Pack items using GOPT policy and map placements back to grocery_items.

    - Runs a single GOPT episode using the configured env/policy.
    - Converts container units to meters based on the provided single box dims.
    - Assigns placements sequentially to grocery_items, setting `obb_target`.

    Notes:
    - GOPT env generates its own sequence of axis-aligned boxes; we map placements to
      grocery items in order. Extents may not exactly match env sizes.
    - If there are fewer placements than items, raises RuntimeError.
    """
    assert len(boxes) == 1, "Only one box size should be provided"
    box_name, box_dims_m = list(boxes.items())[0]

    grocery_item_sizes = [gi.extent for gi in grocery_items]

    # Initialize GOPT packer (paths can be customized)
    packer = BoxPacker(
        config=_abs_config_path("../GOPT/cfg/config.yaml"),
        checkpoint=_abs_config_path(
            "/home/blei/BagBuddy/evaluation/c_evaluator/GOPT/weights/OnlinePack-v1_10-10-10_EMS_80_random_PPO_seed5_Adam_2026.01.19-16-43-06/policy_step_best.pth"
        ),
        device=None,
        box_size_set=grocery_item_sizes,
        container_size = box_dims_m
    )

    result = packer.pack_one_episode(render=render)
    placements = result.boxes

    if len(placements) == 0:
        raise RuntimeError("GOPT produced no placements; cannot map items.")

    # Compute scale: env units -> meters, per axis
    container_units = tuple(packer.args.env.container_size)
    #raise NotImplementedError("Need to use the packers scale factor here. Warning: The packers scale factor is currently an array ...")
    axis_scales = _compute_axis_scales(box_dims_m, container_units)

    # Assign placements to grocery items sequentially
    n_assign = min(len(grocery_items), len(placements))

    for i in range(n_assign):
        gi = grocery_items[i]
        p = placements[i]

        # Convert env position (corner) from units to meters
        pos_units = np.array(p.pos_xyz, dtype=float)
        pos_m_corner = pos_units * axis_scales

        # Use the grocery item's own extent for center offset
        extent_m = np.array(gi.extent, dtype=float)

        # Retrieve rotation from axis swap between expected (scaled) and placed sizes
        R_mat = _retrieve_rotation_matrix(p.size_xyz, packer.groceries_scaled[i])

        center_m = pos_m_corner + p.size_xyz*axis_scales * 0.5

        # Set target OBB on the grocery item
        gi.obb_target = o3d.geometry.OrientedBoundingBox(
            center=center_m, R=R_mat, extent=extent_m
        )

        print(
            f"Item {i:02d}: '{gi.name}' placed at {gi.obb_target.center} m, "
            f"extent {gi.obb_target.extent} m"
        )
    # Any items not assigned are considered unpacked
    if n_assign < len(grocery_items):
        missing = len(grocery_items) - n_assign
        names = [gi.name for gi in grocery_items[n_assign:]]
        raise RuntimeError(f"Packing failed, {missing} items could not be packed: {names}")

    return grocery_items


if __name__ == "__main__":
    # Run the full experiment harness using c_evaluator
    mode_name = "gopt"
    run_exp(
        lambda items, boxes, vis: run_gopt_evaluation(items, boxes, vis, render=False),
        exp_name=mode_name,
        enable_gui=False,
        static=True,
    )

    
