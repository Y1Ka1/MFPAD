# MFPAD: Memory-Forgetting Planning for Long-Horizon End-to-End Autonomous Driving

## Paper Information

- Paper ID: `10554`
- Title: `MFPAD: Memory-Forgetting Planning for Long-Horizon End-to-End Autonomous Driving`
- Authors:
  - `Yikai Wu*`
  - `Qizhou Hu`
  - `Aiguo Lei`
  - `Ziying Song*`
- Affiliations:
  - `Yikai Wu`, `Qizhou Hu`, and `Aiguo Lei`: Nanjing University of Science and Technology, China
  - `Ziying Song`: Beijing Jiaotong University, China
- Corresponding authors: `Yikai Wu` and `Ziying Song`

This repository contains the code, scripts, and dataset/database preparation notes used for the MFPAD project. The method replaces the original MLP trajectory refinement head in planning-oriented end-to-end autonomous driving with a memory-forgetting planning head composed of:

- an LSTM-based memory branch,
- a Transformer-based forgetting branch, and
- a gated fusion module.

The repository includes:

- open-loop training and evaluation code for nuScenes and Adv-nuScenes,
- controlled ablation and alternative-head configurations,
- extended-horizon 12 s analysis utilities,
- NAVSIM helper scripts built around the official NAVSIM v1.1 devkit,
- close-loop Bench2Drive scripts and evaluation utilities,
- dataset/database manifests and reproduction notes.

`MFPAD manuscript.pdf` is included in the repository for reference.

## Reproducibility Coverage

The repository contains scripts and configuration files for the following result groups reported in the paper:

- open-loop nuScenes experiments,
- open-loop Adv-nuScenes robustness experiments,
- controlled ablations (`w/o memory`, `w/o forgetting`),
- alternative heads (`GRU`, `Transformer-only`),
- extended-horizon 12 s analysis,
- closed-loop Bench2Drive experiments.

NAVSIM support in this repository is organized around the official `autonomousvision/navsim` `v1.1` devkit, which is the branch corresponding to the `navtest` benchmark used in the NAVSIM v1 leaderboard.

## Repository Layout

### Top-level files

| Path | Description |
| --- | --- |
| `README.md` | Main project description, paper metadata, repository structure, and reproduction entry points. |
| `MFPAD manuscript.pdf` | PDF version of the paper associated with this repository. |
| `LICENSE` | Repository license. |
| `.gitignore` | Ignore rules for datasets, checkpoints, caches, and local outputs. |
| `docs/DATASETS.md` | Dataset/database manifest: what external data are required, how they are used, and where they should be placed. |
| `docs/REPRODUCTION.md` | Result-oriented reproduction guide mapping paper experiments to scripts, configs, and commands. |

### Main source directories

| Path | Description |
| --- | --- |
| `open_loop/` | Open-loop training, evaluation, data conversion, k-means anchor generation, analysis, and visualization code. |
| `close_loop/` | Close-loop code and external benchmark integrations, mainly for Bench2Drive. |

### Important open-loop files

| Path | Description |
| --- | --- |
| `open_loop/requirement.txt` | Python dependencies for the open-loop pipeline. |
| `open_loop/docs/quick_start.md` | Environment setup notes for open-loop experiments. |
| `open_loop/scripts/create_data.sh` | Generates nuScenes metadata files used by the pipeline. |
| `open_loop/scripts/kmeans.sh` | Generates the detection, map, motion, and planning anchors. |
| `open_loop/run_nuscenes_6s_infos.sh` | Generates 6 s nuScenes info files. |
| `open_loop/run_advnusc_eval.sh` | Evaluates the 3 s model on Adv-nuScenes. |
| `open_loop/run_ablations_6s_train.sh` | Trains the `w/o memory` and `w/o forgetting` variants. |
| `open_loop/run_ablations_6s_eval.sh` | Evaluates the ablation variants on nuScenes. |
| `open_loop/run_ablations_advnusc_eval_6s.sh` | Evaluates the ablation variants on Adv-nuScenes. |
| `open_loop/tools/compute_model_complexity.py` | Reports parameter counts and FLOPs for the main and ablation models. |
| `open_loop/tools/analysis/extended_horizon_summary.py` | Summarizes 7-12 s extended-horizon results into CSV/TXT/TeX outputs. |

### Important open-loop configuration files

| Path | Description |
| --- | --- |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD.py` | Main 3 s MFPAD configuration. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_6s.py` | Main 6 s MFPAD configuration. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_12s.py` | Extended-horizon 12 s evaluation configuration. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_womemory.py` | Ablation without the memory branch. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_woforgetting.py` | Ablation without the forgetting branch. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_gru.py` | GRU-based alternative head. |
| `open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_transonly.py` | Transformer-only alternative head. |

### Important close-loop files

| Path | Description |
| --- | --- |
| `close_loop/quick_start.md` | Environment setup notes for the close-loop pipeline. |
| `close_loop/NAVSIM/` | Helper scripts, adapter templates, and notes for reproducing the NAVSIM experiments with the official NAVSIM v1.1 devkit. |
| `close_loop/VAD_MomAD/Bench2DriveZoo/` | Active Bench2Drive-based close-loop training/evaluation code used by this repository. |
| `close_loop/VAD_MomAD/Bench2DriveZoo/docs/INSTALL.md` | Installation notes for Bench2DriveZoo. |
| `close_loop/VAD_MomAD/Bench2DriveZoo/docs/DATA_PREP.md` | Bench2Drive data preparation guide. |
| `close_loop/VAD_MomAD/Bench2DriveZoo/docs/TRAIN_EVAL.md` | Training and evaluation guide for Bench2DriveZoo. |
| `close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_momad.sh` | Single-model Bench2Drive evaluation script for the MomAD baseline. |
| `close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_mfpad.sh` | Single-model Bench2Drive evaluation script for MFPAD. |
| `close_loop/VAD_MomAD/tools/merge_route_json.py` | Merges route-wise evaluation JSON files. |
| `close_loop/VAD_MomAD/tools/ability_benchmark.py` | Computes Bench2Drive ability metrics. |
| `close_loop/VAD_MomAD/tools/efficiency_smoothness_benchmark.py` | Computes efficiency and smoothness metrics. |

### Auxiliary close-loop directories

| Path | Description |
| --- | --- |
| `close_loop/VAD_MomAD/` | Main close-loop branch used in this release. |
| `close_loop/SparseDrive_MomAD/` | Auxiliary close-loop branch retained for related baseline support and comparison code. |

## Required Datasets / Databases

The experiments in this project use the following datasets/databases:

- `nuScenes` with CAN bus expansion
- `Adv-nuScenes`
- `Bench2Drive Base v1`
- `CARLA 0.9.15 + AdditionalMaps` for close-loop evaluation

The exact expected directory layout, generated metadata files, benchmark asset manifests, and derived files are documented in:

- [docs/DATASETS.md](docs/DATASETS.md)

Official repositories, download entry points, and the accompanying project asset package are also listed there.

## How to Reproduce the Results

The full command-level guide is documented in:

- [docs/REPRODUCTION.md](docs/REPRODUCTION.md)

At a high level:

1. Prepare the open-loop environment with `open_loop/requirement.txt`.
2. Download and organize the required databases listed in `docs/DATASETS.md`.
3. Generate nuScenes metadata and anchors with:
   - `open_loop/scripts/create_data.sh`
   - `open_loop/scripts/kmeans.sh`
   - `open_loop/run_nuscenes_6s_infos.sh`
4. Train or evaluate the main MFPAD models using the configs under `open_loop/projects/configs/`.
5. Run ablations and alternative heads using the dedicated configs and scripts.
6. For NAVSIM, use the helper scripts under `close_loop/NAVSIM/` together with the official NAVSIM `v1.1` devkit.
   - Install the included adapter template into the external NAVSIM checkout with `close_loop/NAVSIM/install_navsim_agent_template.sh`.
7. For Bench2Drive, follow `close_loop/quick_start.md` and `close_loop/VAD_MomAD/Bench2DriveZoo/docs/`.
8. Evaluate close-loop results with:
   - `close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_momad.sh`
   - `close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_mfpad.sh`

## Resource Delivery

- Datasets and benchmark assets are organized through the official sources listed in [docs/DATASETS.md](docs/DATASETS.md).
- Model checkpoints are distributed through the project cloud-drive link/package.
- Generated experiment assets such as metadata files, anchors, and result folders can be prepared with the provided scripts and may also be shared through the same project cloud-drive link/package.
- The reproduction guide identifies the MFPAD-specific scripts and entry points across the open-loop and close-loop pipelines.

## Citation

If you use this repository, please cite the associated MFPAD paper.
