import argparse
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description="KMeans for 12s planning anchors")
    parser.add_argument(
        "--info-path",
        type=str,
        default="data/infos/nuscenes_infos_train_12s.pkl",
        help="Path to 12s training info pkl",
    )
    parser.add_argument("--fut-ts", type=int, default=24, help="Future steps")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters")
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/kmeans/kmeans_plan_6_12s.npy",
        help="Output path for plan anchors",
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
    data_infos = list(data["infos"])

    navi_trajs = [[], [], []]
    for idx in tqdm(range(len(data_infos))):
        info = data_infos[idx]
        plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
        plan_mask = info["gt_ego_fut_masks"]
        cmd = info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1)
        if plan_mask.sum() != args.fut_ts:
            continue
        navi_trajs[cmd].append(plan_traj)

    clusters = []
    os.makedirs(args.plot_dir, exist_ok=True)
    for trajs in navi_trajs:
        if len(trajs) == 0:
            raise RuntimeError("No valid trajectories for planning kmeans.")
        trajs = np.concatenate(trajs, axis=0).reshape(-1, args.fut_ts * 2)
        cluster = KMeans(n_clusters=args.k).fit(trajs).cluster_centers_
        cluster = cluster.reshape(-1, args.fut_ts, 2)
        clusters.append(cluster)
        for j in range(args.k):
            plt.scatter(cluster[j, :, 0], cluster[j, :, 1])
    plt.savefig(os.path.join(args.plot_dir, f"plan_{args.k}_12s"), bbox_inches="tight")
    plt.close()

    clusters = np.stack(clusters, axis=0)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, clusters)


if __name__ == "__main__":
    main()
