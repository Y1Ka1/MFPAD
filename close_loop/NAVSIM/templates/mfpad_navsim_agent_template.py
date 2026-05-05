"""Template MFPAD-to-NAVSIM agent for the official NAVSIM v1.1 devkit.

This file is an integration template rather than a drop-in reproduction model.
It follows the official NAVSIM AbstractAgent interface documented in:
https://github.com/autonomousvision/navsim/blob/v1.1/docs/agents.md

The intent is to give this repository a concrete NAVSIM experiment entry point
and a clean location for the missing MFPAD-specific adapter code. After copying
this file into the official NAVSIM checkout, replace the placeholder methods
with the actual feature construction and checkpoint-loading logic used by your
final MFPAD NAVSIM release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

try:
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataclasses import SensorConfig
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install this file into the official NAVSIM v1.1 checkout before use."
    ) from exc


class MFPADNavsimAgentTemplate(AbstractAgent, nn.Module):
    """Skeleton agent used to port an MFPAD checkpoint into NAVSIM."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        mfpad_repo_root: Optional[str] = None,
        mfpad_config_path: Optional[str] = None,
        device: str = "cuda",
        use_all_sensors: bool = True,
        trajectory_horizon_s: float = 4.0,
        trajectory_num_poses: int = 40,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.mfpad_repo_root = mfpad_repo_root
        self.mfpad_config_path = mfpad_config_path
        self.device_name = device
        self.use_all_sensors = use_all_sensors
        self.trajectory_horizon_s = trajectory_horizon_s
        self.trajectory_num_poses = trajectory_num_poses

        # Replace this placeholder with the actual MFPAD/NAVSIM backbone.
        self.model = nn.Identity()

    def name(self) -> str:
        return "mfpad_navsim_agent_template"

    def initialize(self) -> None:
        """Load the MFPAD checkpoint inside the NAVSIM worker process."""
        if not self.checkpoint_path:
            return

        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        self._load_mfpad_checkpoint(checkpoint)
        self.to(torch.device(self.device_name))
        self.eval()

    def get_sensor_config(self) -> SensorConfig:
        """Return the sensor request used by the adapter."""
        if self.use_all_sensors:
            return SensorConfig.build_all_sensors()

        raise NotImplementedError(
            "Define a reduced SensorConfig here if you want a lighter NAVSIM setup."
        )

    def get_feature_builders(self) -> List[Any]:
        raise NotImplementedError(
            "Create NAVSIM feature builders that convert AgentInput into the "
            "features consumed by your MFPAD adapter."
        )

    def get_target_builders(self) -> List[Any]:
        raise NotImplementedError(
            "Create NAVSIM target builders for supervised training in NAVSIM."
        )

    def forward(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Implement the MFPAD forward pass and return a dict containing "
            "'trajectory' with shape [B, T, 3]."
        )

    def compute_loss(
        self,
        features: Dict[str, Any],
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Implement the NAVSIM-side trajectory loss and any auxiliary losses."
        )

    def get_optimizers(self) -> Any:
        raise NotImplementedError(
            "Return the optimizer or optimizer/scheduler dict for NAVSIM training."
        )

    def get_training_callbacks(self) -> List[Any]:
        return []

    def _load_mfpad_checkpoint(self, checkpoint: Path) -> None:
        """Map the published MFPAD weights into the NAVSIM agent.

        Typical integration steps:
        1. Build the MFPAD model from `mfpad_config_path`.
        2. Load the checkpoint state dict.
        3. Attach or wrap the planning head so NAVSIM receives a trajectory in
           local coordinates.
        4. Resample the trajectory if your published MFPAD head does not
           natively output NAVSIM's 4 s / 10 Hz evaluation horizon.
        """
        raise NotImplementedError(
            f"Checkpoint loading hook is still a template: {checkpoint}"
        )
