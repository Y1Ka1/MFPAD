import argparse
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = -boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum("aij,jka->aik", trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new


def parse_args():
    parser = argparse.ArgumentParser(description="KMeans for 12s motion anchors")
    parser.add_argument(
        "--info-path",
        type=str,
        default="data/infos/nuscenes_infos_train_12s.pkl",
        help="Path to 12s training info pkl",
    )
    parser.add_argument("--fut-ts", type=int, default=24, help="Future steps")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters")
    parser.add_argument("--dist-thresh", type=float, default=55, help="Distance filter")
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/kmeans/kmeans_motion_6_12s.npy",
        help="Output path for motion anchors",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="vis/kmeans",
        help="Directory for optional plot output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = mmcv.load(args.info_path)
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    intention = {i: [] for i in range(len(CLASSES))}

    for idx in tqdm(range(len(data_infos))):
        info = data_infos[idx]
        boxes = info["gt_boxes"]
        names = info["gt_names"]
        fut_masks = info["gt_agent_fut_masks"]
        trajs = info["gt_agent_fut_trajs"]
        velos = info["gt_velocity"]
        labels = []
        for cat in names:
            if cat in CLASSES:
                labels.append(CLASSES.index(cat))
            else:
                labels.append(-1)
        labels = np.array(labels)
        if len(boxes) == 0:
            continue
        for i in range(len(CLASSES)):
            cls_mask = labels == i
            box_cls = boxes[cls_mask]
            fut_masks_cls = fut_masks[cls_mask]
            trajs_cls = trajs[cls_mask]
            velos_cls = velos[cls_mask]

            distance = np.linalg.norm(box_cls[:, :2], axis=1)
            mask = np.logical_and(
                fut_masks_cls.sum(axis=1) == args.fut_ts,
                distance < args.dist_thresh,
            )
            trajs_cls = trajs_cls[mask]
            box_cls = box_cls[mask]
            velos_cls = velos_cls[mask]

            trajs_agent = lidar2agent(trajs_cls, box_cls)
            if trajs_agent.shape[0] == 0:
                continue
            intention[i].append(trajs_agent)

    clusters = []
    os.makedirs(args.plot_dir, exist_ok=True)
    for i in range(len(CLASSES)):
        if len(intention[i]) == 0:
            raise RuntimeError(f"No valid trajectories for class {CLASSES[i]}")
        intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, args.fut_ts * 2)
        if intention_cls.shape[0] < args.k:
            raise RuntimeError(f"Not enough samples for class {CLASSES[i]}: {intention_cls.shape[0]}")
        cluster = KMeans(n_clusters=args.k).fit(intention_cls).cluster_centers_
        cluster = cluster.reshape(-1, args.fut_ts, 2)
        clusters.append(cluster)
        for j in range(args.k):
            plt.scatter(cluster[j, :, 0], cluster[j, :, 1])
        plt.savefig(
            os.path.join(args.plot_dir, f"motion_intention_{CLASSES[i]}_{args.k}_12s"),
            bbox_inches="tight",
        )
        plt.close()

    clusters = np.stack(clusters, axis=0)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, clusters)


if __name__ == "__main__":
    main()
