# Reproduction Guide

This file maps the major result groups in the paper to the scripts, configurations, and commands provided in the repository.

## 1. Environment setup

### 1.1 Open-loop

```bash
conda create -n mfpad python=3.8 -y
conda activate mfpad
pip install -r open_loop/requirement.txt
```

Additional notes:

- `open_loop/docs/quick_start.md`

### 1.2 Close-loop

Follow:

- `close_loop/quick_start.md`
- `close_loop/VAD_MomAD/Bench2DriveZoo/docs/INSTALL.md`
- `close_loop/VAD_MomAD/Bench2DriveZoo/docs/DATA_PREP.md`
- `close_loop/VAD_MomAD/Bench2DriveZoo/docs/TRAIN_EVAL.md`

## 2. Dataset and database preparation

See:

- [DATASETS.md](DATASETS.md)

Open-loop data preparation:

```bash
cd open_loop
bash scripts/create_data.sh
bash scripts/kmeans.sh
bash run_nuscenes_6s_infos.sh
```

Extended-horizon 12 s data preparation:

```bash
cd open_loop
PYTHONPATH=. python3 tools/data_converter/nuscenes_converter_12s.py nuscenes \
  --root-path ./data/nuscenes \
  --canbus ./data/nuscenes \
  --out-dir ./data/infos/ \
  --extra-tag nuscenes \
  --version v1.0

PYTHONPATH=. python3 tools/kmeans/kmeans_motion_12s.py
PYTHONPATH=. python3 tools/kmeans/kmeans_plan_12s.py
```

## 3. Main open-loop MFPAD models

### 3.1 Train the 3 s model

```bash
cd open_loop
bash ./tools/dist_train.sh \
  projects/configs/MomAD_small_stage2_roboAD.py \
  8 \
  --deterministic
```

### 3.2 Train the 6 s model

```bash
cd open_loop
bash ./tools/dist_train.sh \
  projects/configs/MomAD_small_stage2_roboAD_6s.py \
  8 \
  --deterministic
```

### 3.3 Evaluate on nuScenes

3 s:

```bash
cd open_loop
bash ./tools/dist_test.sh \
  projects/configs/MomAD_small_stage2_roboAD.py \
  /path/to/mfpad_3s_checkpoint.pth \
  1 \
  --deterministic \
  --eval bbox
```

6 s:

```bash
cd open_loop
bash ./tools/dist_test.sh \
  projects/configs/MomAD_small_stage2_roboAD_6s.py \
  /path/to/mfpad_6s_checkpoint.pth \
  1 \
  --deterministic \
  --eval bbox
```

## 4. Adv-nuScenes robustness experiments

### 4.1 Evaluate the 3 s model on Adv-nuScenes

```bash
cd open_loop
CKPT_PATH=/path/to/mfpad_3s_checkpoint.pth \
bash run_advnusc_eval.sh
```

### 4.2 Evaluate the 6 s model on Adv-nuScenes

```bash
cd open_loop
MOMAD_USE_ALL_SCENES=1 \
MOMAD_NUSC_ROOT=data/advnusc/ \
PYTHONPATH=. python3 tools/test.py \
  projects/configs/MomAD_small_stage2_roboAD_6s.py \
  /path/to/mfpad_6s_checkpoint.pth \
  --eval bbox \
  --cfg-options \
    data.test.data_root=data/advnusc/ \
    data.val.data_root=data/advnusc/ \
    data.test.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.val.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.test.eval_config.data_root=data/advnusc/ \
    data.val.eval_config.data_root=data/advnusc/ \
    data.test.eval_config.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.val.eval_config.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    evaluation.eval_mode.with_planning=True \
    evaluation.eval_mode.with_motion=False \
    evaluation.eval_mode.with_det=False \
    evaluation.eval_mode.with_map=False \
    evaluation.eval_mode.with_tracking=False
```

## 5. Ablation studies

### 5.1 Train `w/o memory` and `w/o forgetting`

```bash
cd open_loop
bash run_ablations_6s_train.sh
```

### 5.2 Evaluate ablations on nuScenes

```bash
cd open_loop
bash run_ablations_6s_eval.sh
```

### 5.3 Evaluate ablations on Adv-nuScenes

```bash
cd open_loop
bash run_ablations_advnusc_eval_6s.sh
```

## 6. Alternative heads

### 6.1 GRU-based head

Config:

```text
open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_gru.py
```

Training:

```bash
cd open_loop
bash ./tools/dist_train.sh \
  projects/configs/MomAD_small_stage2_roboAD_6s_gru.py \
  8 \
  --deterministic
```

### 6.2 Transformer-only head

Config:

```text
open_loop/projects/configs/MomAD_small_stage2_roboAD_6s_transonly.py
```

Training:

```bash
cd open_loop
bash ./tools/dist_train.sh \
  projects/configs/MomAD_small_stage2_roboAD_6s_transonly.py \
  8 \
  --deterministic
```

## 7. Extended-horizon 12 s analysis

### 7.1 Run 12 s evaluation

```bash
cd open_loop
bash ./tools/dist_test.sh \
  projects/configs/MomAD_small_stage2_roboAD_12s.py \
  /path/to/mfpad_6s_checkpoint_or_12s_checkpoint.pth \
  1 \
  --deterministic \
  --eval bbox
```

### 7.2 Summarize the extended-horizon results

```bash
cd open_loop
PYTHONPATH=. python3 tools/analysis/extended_horizon_summary.py \
  --sparsedrive-json /path/to/sparsedrive_12s_metrics.json \
  --momad-json /path/to/momad_12s_metrics.json \
  --mfpad-json /path/to/mfpad_12s_metrics.json \
  --out-csv /path/to/out/extended_horizon.csv \
  --out-txt /path/to/out/extended_horizon.txt \
  --out-tex /path/to/out/extended_horizon.tex
```

## 8. Model complexity and FLOPs

Use:

```bash
cd open_loop
PYTHONPATH=. python3 tools/compute_model_complexity.py
```

Single variant:

```bash
cd open_loop
PYTHONPATH=. python3 tools/compute_model_complexity.py --variant mfpad
```

If you also want to compare against an external original MomAD checkout, set:

```bash
export MOMAD_ORIG_ROOT=/path/to/MomAD-original
```

## 9. Closed-loop Bench2Drive experiments

### 9.1 Train the close-loop baseline and MFPAD models

MomAD baseline:

```bash
cd close_loop/VAD_MomAD/Bench2DriveZoo
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/MomAD_base_e2e_b2d.py 1
```

MFPAD close-loop experiments use the active VAD_MomAD code path in this repository together with the shared MFPAD checkpoint link/package, and evaluation is performed through `run_eval_b2d_single_mfpad.sh`.

```bash
cd close_loop/VAD_MomAD/Bench2DriveZoo
./adzoo/vad/dist_train.sh ./adzoo/vad/configs/VAD/MomAD_base_e2e_b2d.py 1
```

### 9.2 Evaluate on Bench2Drive

MomAD:

```bash
CKPT_PATH=/path/to/momad_b2d_checkpoint.pth \
bash close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_momad.sh
```

MFPAD:

```bash
CKPT_PATH=/path/to/mfpad_b2d_checkpoint.pth \
bash close_loop/VAD_MomAD/leaderboard/scripts/run_eval_b2d_single_mfpad.sh
```

### 9.3 Aggregate close-loop metrics

```bash
cd close_loop/VAD_MomAD
python tools/merge_route_json.py -f /path/to/json_folder
python tools/ability_benchmark.py -r merge.json
python tools/efficiency_smoothness_benchmark.py -f merge.json -m /path/to/metric_out_dir
```

## 10. NAVSIM experiments

The paper references NAVSIM v1 metrics on the `navtest` benchmark. The current repository now includes helper scripts under `close_loop/NAVSIM/` that target the official NAVSIM `v1.1` devkit.

### 10.1 Prepare the official NAVSIM workspace

Follow the official resources:

- `https://github.com/autonomousvision/navsim/tree/v1.1`
- `https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md`

Then set:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### 10.2 Download NAVSIM data

```bash
bash close_loop/NAVSIM/run_navsim_download.sh
```

Optional targets:

```bash
NAVSIM_DOWNLOAD_TARGETS="maps navtrain test"
```

### 10.3 Build the metric cache

```bash
bash close_loop/NAVSIM/run_navsim_metric_caching.sh
```

### 10.4 Install the included NAVSIM adapter template

The repository includes an MFPAD-to-NAVSIM adapter template derived from the official NAVSIM agent interface described in `docs/agents.md`.

Install it into the external NAVSIM devkit checkout:

```bash
bash close_loop/NAVSIM/install_navsim_agent_template.sh
```

This copies:

- `close_loop/NAVSIM/templates/mfpad_navsim_agent_template.py`
- `close_loop/NAVSIM/templates/mfpad_navsim_agent_template.yaml`

into the expected NAVSIM v1.1 locations under `${NAVSIM_DEVKIT_ROOT}`.

### 10.5 Train a NAVSIM-compatible agent

This wrapper follows the official NAVSIM training entry point:

```bash
NAVSIM_AGENT=mfpad_navsim_agent_template \
TRAIN_TEST_SPLIT=navtrain \
bash close_loop/NAVSIM/run_navsim_train.sh
```

The included adapter template provides the NAVSIM experiment entry point and the expected file layout for loading the shared MFPAD checkpoints from the project cloud-drive link/package inside the official NAVSIM v1.1 workflow.

### 10.6 Evaluate on `navtest`

```bash
NAVSIM_AGENT=constant_velocity_agent \
TRAIN_TEST_SPLIT=navtest \
bash close_loop/NAVSIM/run_navsim_eval.sh
```

For an MFPAD checkpoint from the project cloud-drive link/package:

```bash
NAVSIM_AGENT=mfpad_navsim_agent_template \
CKPT_PATH=/path/to/checkpoint.ckpt \
TRAIN_TEST_SPLIT=navtest \
bash close_loop/NAVSIM/run_navsim_eval.sh
```

### 10.7 Submission and leaderboard

Use the official NAVSIM v1.1 submission instructions:

- `https://github.com/autonomousvision/navsim/blob/v1.1/docs/submission.md`

Official leaderboard:

- `https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim`

## 11. Result groups covered by the repository

Covered directly by the current repository:

- nuScenes open-loop experiments
- Adv-nuScenes robustness experiments
- ablation experiments
- GRU / Transformer-only alternative heads
- extended-horizon 12 s analysis
- NAVSIM helper scripts, adapter templates, and reproduction wrappers
- Bench2Drive close-loop experiments
