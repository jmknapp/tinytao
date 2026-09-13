"""Run directory layout.

results/runs/<timestamp>_<name>/
    config.yaml
    meta.json
    metrics/final.pt          # per-network final scalars
    metrics/trajectories.pt   # [P, T] time series (not every optimizer step)
    metrics/summary.json
    checkpoints/successful.pt # parameters of exact solvers (capped)
    checkpoints/representatives.pt
    figures/

MySQL tinytao.nets (credentials from MYSQL_* env): exact nets, 390 params + classification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.config import ExperimentConfig


def new_run_dir(root: str | Path, run_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(root) / "results" / "runs" / f"{stamp}_{run_name}"
    for sub in ("metrics", "checkpoints", "figures", "logs"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_yaml(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _json_default(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(f"cannot json-encode {type(obj)}")


def save_run_skeleton(run_dir: Path, config: ExperimentConfig, meta: dict[str, Any]) -> None:
    config.to_yaml(run_dir / "config.yaml")
    write_json(run_dir / "meta.json", meta)


def save_metrics(run_dir: Path, payload: dict[str, Any], trajectories: dict[str, torch.Tensor]) -> None:
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    torch.save(payload, run_dir / "metrics" / "final.pt")
    cpu = {k: v.detach().cpu() for k, v in trajectories.items()}
    torch.save(cpu, run_dir / "metrics" / "trajectories.pt")


def save_checkpoints(
    run_dir: Path,
    successful: dict[str, torch.Tensor] | None,
    representatives: dict[str, torch.Tensor] | None,
) -> None:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if successful is not None:
        torch.save(successful, run_dir / "checkpoints" / "successful.pt")
    if representatives is not None:
        torch.save(representatives, run_dir / "checkpoints" / "representatives.pt")
