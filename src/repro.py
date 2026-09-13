"""Capture everything needed to replay a run."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import torch


def _git_commit(repo: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_record(repo: str | None = None) -> dict[str, Any]:
    gpu = None
    cuda = None
    if torch.cuda.is_available():
        cuda = torch.version.cuda
        idx = torch.cuda.current_device()
        gpu = {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(idx)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(idx).total_memory),
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": cuda,
        "gpu": gpu,
        "git_commit": _git_commit(repo) if repo else None,
        "cwd": os.getcwd(),
    }


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
