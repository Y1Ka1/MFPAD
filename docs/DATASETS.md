# Dataset and Database Manifest

This file lists the external datasets/databases, generated metadata files, and benchmark assets required to reproduce the experiments in the MFPAD paper.

For each dataset/database below, we list:

- the official repository,
- the official download entry point,
- the local structure expected by this repository.

## 1. Open-loop datasets

### 1.1 nuScenes

Used for:

- main open-loop results,
- short-horizon (3 s) experiments,
- long-horizon (6 s) experiments,
- extended-horizon (12 s) analysis,
- ablations and alternative heads.

Expected location:

```text
open_loop/data/nuscenes/
```

Required external content:

- official nuScenes dataset
- official CAN bus expansion

Official repository:

- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)

Official download entry points:

- [nuScenes download page](https://www.nuscenes.org/download)
- [nuScenes devkit setup and CAN bus expansion instructions](https://github.com/nutonomy/nuscenes-devkit)

Generated metadata files:

```text
open_loop/data/infos/nuscenes_infos_train.pkl
open_loop/data/infos/nuscenes_infos_val.pkl
open_loop/data/infos/nuscenes_infos_train_6s.pkl
open_loop/data/infos/nuscenes_infos_val_6s.pkl
open_loop/data/infos/nuscenes_infos_train_12s.pkl
open_loop/data/infos/nuscenes_infos_val_12s.pkl
```

How to generate:

- `open_loop/scripts/create_data.sh`
- `open_loop/run_nuscenes_6s_infos.sh`
- `open_loop/tools/data_converter/nuscenes_converter_12s.py`

### 1.2 Adv-nuScenes

Used for:

- robustness experiments on adversarially perturbed scenes.

Expected location:

```text
open_loop/data/advnusc/
open_loop/data/infos_advnusc/advnusc_infos_val.pkl
```

Required external content:

- Adv-nuScenes data generated from the Challenger framework or the equivalent official release used in the paper.

Official repository:

- [Challenger: Affordable Adversarial Driving Video Generation](https://github.com/Pixtella/Challenger)

Official project page:

- [Challenger project page](https://pixtella.github.io/Challenger/)

Download / generation entry point:

- follow the official Challenger repository instructions to generate or obtain the adversarial nuScenes-style data used for Adv-nuScenes evaluation.

### 1.3 Generated anchors

Used for:

- detection,
- map prediction,
- motion prediction,
- planning.

Expected location:

```text
open_loop/data/kmeans/kmeans_det_900.npy
open_loop/data/kmeans/kmeans_map_100.npy
open_loop/data/kmeans/kmeans_motion_6.npy
open_loop/data/kmeans/kmeans_plan_6.npy
open_loop/data/kmeans/kmeans_motion_6_12s.npy
open_loop/data/kmeans/kmeans_plan_6_12s.npy
```

How to generate:

- `open_loop/scripts/kmeans.sh`
- `open_loop/tools/kmeans/kmeans_motion_12s.py`
- `open_loop/tools/kmeans/kmeans_plan_12s.py`

## 2. Close-loop datasets and simulator assets

### 2.1 Bench2Drive Base v1

Used for:

- close-loop training and evaluation on Bench2Drive.

Expected location:

```text
close_loop/VAD_MomAD/Bench2DriveZoo/data/bench2drive/
```

Required external content:

- Bench2Drive Base v1 dataset
- official route/evaluation assets used by Bench2Drive

Official repository:

- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

Official download entry points:

- [Bench2Drive Base on Hugging Face](https://huggingface.co/datasets/rethinklab/Bench2Drive)
- [Bench2Drive Full on Hugging Face](https://huggingface.co/datasets/rethinklab/Bench2Drive-Full)
- [Bench2Drive Full-Supplement on Hugging Face](https://huggingface.co/datasets/rethinklab/Bench2Drive-Full-Sup)

Useful manifests already included in the repository:

```text
close_loop/VAD_MomAD/docs/bench2drive_base_1000.json
close_loop/VAD_MomAD/docs/bench2drive_mini_10.json
close_loop/VAD_MomAD/docs/bench2drive_full+sup_13638.json
```

Required generated metadata files:

```text
close_loop/VAD_MomAD/Bench2DriveZoo/data/infos/b2d_infos_train.pkl
close_loop/VAD_MomAD/Bench2DriveZoo/data/infos/b2d_infos_val.pkl
close_loop/VAD_MomAD/Bench2DriveZoo/data/infos/b2d_map_infos.pkl
```

Required anchor files:

```text
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_det_900.npy
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_map_100.npy
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_motion_6.npy
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_plan_1.npy
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_plan_3.npy
close_loop/VAD_MomAD/Bench2DriveZoo/data/kmeans/kmeans_plan_6.npy
```

### 2.2 CARLA

Used for:

- Bench2Drive close-loop evaluation.

Required external content:

- CARLA `0.9.15`
- AdditionalMaps for CARLA `0.9.15`

Expected environment variable:

```bash
export CARLA_ROOT=/path/to/carla
```

Relevant helper script:

```text
close_loop/VAD_MomAD/Bench2DriveZoo/scripts/download_carla_0_9_15.sh
```

Official benchmark repository:

- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

Official simulator repository:

- [CARLA official repository](https://github.com/carla-simulator/carla)

Official simulator download entry point:

- [CARLA 0.9.15 release with `CARLA_0.9.15` and `AdditionalMaps_0.9.15`](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

## 2.3 NAVSIM / OpenScene / nuPlan maps

Used for:

- NAVSIM `navtest` experiments reported in the paper.

Official repository:

- [NAVSIM official repository](https://github.com/autonomousvision/navsim)

Important branch for this paper:

- [NAVSIM `v1.1` branch / release line](https://github.com/autonomousvision/navsim/tree/v1.1)

Official documentation:

- [NAVSIM v1.1 README](https://github.com/autonomousvision/navsim/blob/v1.1/README.md)
- [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md)
- [NAVSIM split documentation](https://github.com/autonomousvision/navsim/blob/v1.1/docs/splits.md)
- [NAVSIM submission guide](https://github.com/autonomousvision/navsim/blob/v1.1/docs/submission.md)

Official download entry points:

- [download_navtrain.sh](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_navtrain.sh)
- [download_test.sh](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_test.sh)
- [OpenScene dataset on Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene)
- [nuPlan maps download index](https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/index.html)

Related official repositories:

- [nuPlan devkit](https://github.com/motional/nuplan-devkit)
- [CARLA official repository](https://github.com/carla-simulator/carla)

Expected external workspace structure for NAVSIM:

```text
<navsim-workspace>/
  navsim/                  # official NAVSIM devkit checkout (v1.1)
  dataset/
    maps/
    navsim_logs/
      mini/
      trainval/
      test/
      private_test_e2e/
    sensor_blobs/
      mini/
      trainval/
      test/
      private_test_e2e/
  exp/
```

Expected environment variables:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

## 3. Checkpoints and pretrained weights

Checkpoints and pretrained weights are delivered through the project cloud-drive link/package.

### 3.1 Backbone / initialization weights

```text
open_loop/ckpt/resnet50-19c8e357.pth
close_loop/VAD_MomAD/Bench2DriveZoo/ckpts/momad_small_b2d_stage1.pth
```

### 3.2 Trained experiment checkpoints

The main experiment checkpoints are shared through the project cloud-drive link/package and correspond to the training outputs under:

```text
open_loop/work_dirs/
close_loop/VAD_MomAD/Bench2DriveZoo/work_dirs/
```

## 4. Project asset package

In addition to the code repository, the project is accompanied by a cloud-drive link/package that can be used to organize:

- trained model checkpoints (`.pth`, `.pt`, `.ckpt`)
- generated `.pkl` metadata files
- generated `.npy` anchors
- evaluation outputs
- visualization folders

Raw datasets and simulator assets should be prepared from the official sources listed above so that the directory layout matches the paths used by the scripts in this repository.

## 5. Related setup documents

For command-level reproduction steps, see:

- [REPRODUCTION.md](REPRODUCTION.md)

For environment setup details, see:

- `open_loop/docs/quick_start.md`
- `close_loop/quick_start.md`
- `close_loop/VAD_MomAD/Bench2DriveZoo/docs/INSTALL.md`
- `close_loop/VAD_MomAD/Bench2DriveZoo/docs/DATA_PREP.md`
