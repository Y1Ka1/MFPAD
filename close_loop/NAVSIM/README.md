# NAVSIM Experiment Helpers

This directory contains helper scripts for reproducing the NAVSIM experiments associated with the MFPAD paper.

These scripts are designed to work with the official NAVSIM devkit rather than replacing it.

## Official NAVSIM resources

- Official repository: [autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- NAVSIM v1.1 branch: [github.com/autonomousvision/navsim/tree/v1.1](https://github.com/autonomousvision/navsim/tree/v1.1)
- Installation guide: [docs/install.md](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md)
- Agent interface guide: [docs/agents.md](https://github.com/autonomousvision/navsim/blob/v1.1/docs/agents.md)
- Splits guide: [docs/splits.md](https://github.com/autonomousvision/navsim/blob/v1.1/docs/splits.md)
- Submission guide: [docs/submission.md](https://github.com/autonomousvision/navsim/blob/v1.1/docs/submission.md)

The `v1.1` branch is the relevant release line for the `navtest` benchmark referenced by the NAVSIM v1 leaderboard.

## What is included here

| File | Purpose |
| --- | --- |
| `run_navsim_download.sh` | Calls the official NAVSIM download scripts for maps and selected dataset splits. |
| `run_navsim_metric_caching.sh` | Runs the official NAVSIM metric cache generation script. |
| `run_navsim_train.sh` | Wrapper around the official NAVSIM training entry point. |
| `run_navsim_eval.sh` | Wrapper around the official NAVSIM local PDM-score evaluation entry point. |
| `install_navsim_agent_template.sh` | Installs the included MFPAD-to-NAVSIM adapter template into an external NAVSIM v1.1 checkout. |
| `templates/mfpad_navsim_agent_template.py` | Template agent class following the official NAVSIM `AbstractAgent` interface. |
| `templates/mfpad_navsim_agent_template.yaml` | Matching Hydra agent config for the template agent. |

## Required environment variables

The scripts assume the same environment variables used by the official NAVSIM installation guide:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

## Workflow organization

These scripts are designed to work together with:

- the official NAVSIM v1.1 devkit,
- the NAVSIM/OpenScene data,
- the nuPlan maps, and
- the project cloud-drive link/package for MFPAD checkpoints.

The included adapter template provides the NAVSIM-side integration entry point used to load and organize MFPAD checkpoints within the official NAVSIM workflow.

## Typical workflow

```bash
bash close_loop/NAVSIM/run_navsim_download.sh
bash close_loop/NAVSIM/run_navsim_metric_caching.sh
bash close_loop/NAVSIM/install_navsim_agent_template.sh
bash close_loop/NAVSIM/run_navsim_train.sh
bash close_loop/NAVSIM/run_navsim_eval.sh
```
