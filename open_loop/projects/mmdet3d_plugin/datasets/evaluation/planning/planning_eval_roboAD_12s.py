from tqdm import tqdm
import csv
import os
import torch
import numpy as np
from shapely.geometry import Polygon

from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


def _get_cfg_value(cfg, key, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _calc_future_steps(future_horizon_sec: float, step_sec: float) -> int:
    steps = int(round(future_horizon_sec / step_sec))
    if steps <= 0:
        raise ValueError(f"future_horizon_sec must be > 0, got {future_horizon_sec}")
    if abs(steps * step_sec - future_horizon_sec) > 1e-6:
        raise ValueError(
            f"future_horizon_sec must be divisible by step_sec, got {future_horizon_sec} and {step_sec}"
        )
    return steps


def _seconds_to_indices(future_horizon_sec: float, step_sec: float):
    sec_max = int(round(future_horizon_sec))
    seconds = list(range(1, sec_max + 1))
    indices = []
    for sec in seconds:
        idx = int(round(sec / step_sec)) - 1
        indices.append(idx)
    return seconds, indices


def check_collision(ego_box, boxes):
    if boxes.shape[0] == 0:
        return False
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly = Polygon([(point[0], point[1]) for point in corners_box[i]])
        if ego_poly.intersects(box_poly):
            return True
    return False


def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2
    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0] + 1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]


class PlanningMetric:
    def __init__(self, n_future: int):
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.Consist = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)
        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)
        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)
        assert B == 1, "only support bs=1"
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()
        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        return torch.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

    def compute_Consist(self, trajs, last_final_planning, gt_trajs_mask):
        return torch.sqrt(
            (((trajs[:, :, :2] - last_final_planning[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes, last_final_planning):
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        Consist = self.compute_Consist(trajs, last_final_planning, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:, :, :2], gt_trajs[:, :, :2], fut_boxes)
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.Consist += Consist.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "L2": self.L2 / self.total,
            "Consist": self.Consist / self.total,
        }


def _write_per_horizon_csv(path, l2_map, coll_map):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seconds = sorted(l2_map.keys(), key=lambda x: int(x[:-1]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + seconds)
        writer.writerow(["L2"] + [f"{l2_map[s]:.6f}" for s in seconds])
        writer.writerow(["collision"] + [f"{coll_map[s]:.6f}" for s in seconds])


def planning_eval_roboAD_12s(results, eval_config, logger):
    future_horizon_sec = _get_cfg_value(eval_config, "future_horizon_sec", 12.0)
    step_sec = _get_cfg_value(eval_config, "step_sec", 0.5)
    n_future = _calc_future_steps(future_horizon_sec, step_sec)

    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False
    )
    planning_metrics = PlanningMetric(n_future=n_future)
    last_final_planning = torch.zeros([1, n_future, 2])

    valid_samples = 0
    skipped_incomplete = 0
    skipped_fut_boxes = 0

    for i, data in enumerate(tqdm(dataloader)):
        sdc_planning = data["gt_ego_fut_trajs"].cumsum(dim=-2).unsqueeze(1)
        sdc_planning_mask = data["gt_ego_fut_masks"].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        fut_boxes = data["fut_boxes"]

        if not sdc_planning_mask[..., :n_future, :].all():
            skipped_incomplete += 1
            continue
        if len(fut_boxes) < n_future:
            skipped_fut_boxes += 1
            continue

        res = results[i]
        pred_sdc_traj = res["img_bbox"]["final_planning"].unsqueeze(0)
        planning_metrics.update(
            pred_sdc_traj[:, :n_future, :2],
            sdc_planning[0, :, :n_future, :2],
            sdc_planning_mask[0, :, :n_future, :2],
            fut_boxes,
            last_final_planning,
        )
        last_final_planning = pred_sdc_traj[:, :n_future, :2]
        valid_samples += 1

    planning_results = planning_metrics.compute()
    seconds, indices = _seconds_to_indices(future_horizon_sec, step_sec)
    l2_vals = planning_results["L2"].tolist()
    coll_vals = planning_results["obj_box_col"].tolist()
    obj_col_vals = planning_results["obj_col"].tolist()
    consist_vals = planning_results["Consist"].tolist()

    l2_per_sec = {f"{sec}s": float(l2_vals[idx]) for sec, idx in zip(seconds, indices)}
    coll_per_sec = {f"{sec}s": float(coll_vals[idx]) for sec, idx in zip(seconds, indices)}

    avg_l2_7_12 = np.mean([l2_per_sec[f"{s}s"] for s in range(7, 13)])
    avg_coll_7_12 = np.mean([coll_per_sec[f"{s}s"] for s in range(7, 13)])

    l2_6 = l2_per_sec["6s"]
    l2_12 = l2_per_sec["12s"]
    coll_6 = coll_per_sec["6s"]
    coll_12 = coll_per_sec["12s"]

    rel_l2 = (l2_12 - l2_6) / l2_6 * 100 if l2_6 > 0 else float("inf")
    rel_coll = (coll_12 - coll_6) / coll_6 * 100 if coll_6 > 0 else float("inf")

    if l2_12 < l2_6:
        print_log("[12s] L2_12s < L2_6s, please inspect.", logger=logger)
    if coll_12 < coll_6:
        print_log("[12s] collision_12s < collision_6s, please inspect.", logger=logger)

    output_dir = _get_cfg_value(eval_config, "eval_output_dir", "./work_dirs/extended_horizon_12s")
    output_prefix = _get_cfg_value(eval_config, "eval_output_prefix", "momad_12s")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{output_prefix}_per_horizon.json")
    per_horizon_csv = os.path.join(output_dir, f"{output_prefix}_per_horizon.csv")

    json_payload = {
        "L2": l2_per_sec,
        "collision": coll_per_sec,
        "obj_col": {f"{sec}s": float(obj_col_vals[idx]) for sec, idx in zip(seconds, indices)},
        "consist": {f"{sec}s": float(consist_vals[idx]) for sec, idx in zip(seconds, indices)},
        "avg_l2_7_12": float(avg_l2_7_12),
        "avg_collision_7_12": float(avg_coll_7_12),
        "rel_l2_6_to_12_pct": float(rel_l2),
        "rel_collision_6_to_12_pct": float(rel_coll),
        "valid_samples": int(valid_samples),
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_fut_boxes": int(skipped_fut_boxes),
        "future_horizon_sec": float(future_horizon_sec),
        "step_sec": float(step_sec),
    }

    import mmcv
    mmcv.dump(json_payload, json_path)
    _write_per_horizon_csv(per_horizon_csv, l2_per_sec, coll_per_sec)

    print_log(
        f"[12s] valid_samples={valid_samples}, skipped_incomplete={skipped_incomplete}, skipped_fut_boxes={skipped_fut_boxes}",
        logger=logger,
    )
    return {
        "avg_l2_7_12": avg_l2_7_12,
        "avg_collision_7_12": avg_coll_7_12,
        "rel_l2_6_to_12_pct": rel_l2,
        "rel_collision_6_to_12_pct": rel_coll,
    }
