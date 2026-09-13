"""Experiment configuration. YAML in, dataclass out."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    name: str = "modular_multiplication"
    p: int = 13
    split: str = "random"
    train_frac: float = 0.7
    split_seed: int = 0
    held_operands: list[int] = field(default_factory=lambda: [-1])
    held_products: list[int] = field(default_factory=lambda: [0])
    cut: int | None = None


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [8])
    activation: str = "relu"
    use_bias: bool = True
    init: str = "xavier_uniform"
    init_scale: float = 1.0
    interaction: str = "add"  # "add" (A), "product" (B), "complex" (C), "tropical" (T), "kirchhoff" (K)


@dataclass
class PopulationConfig:
    size: int = 2048
    min_size: int = 64


@dataclass
class TrainConfig:
    epochs: int = 2000
    batch_size: int = 0  # 0 = full training set each step
    lr: float = 1.0e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 10
    eval_every: int = 10
    grad_clip: float | None = None


@dataclass
class SweepConfig:
    hidden_widths: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 12, 16, 24, 32])
    protocol: str = "full_domain"
    depths: list[int] = field(default_factory=lambda: [1])


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    seed: int = 0
    device: str = "cuda"
    run_name: str = "phase1"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(
            task=TaskConfig(**data.get("task", {})),
            model=ModelConfig(**data.get("model", {})),
            population=PopulationConfig(**data.get("population", {})),
            train=TrainConfig(**data.get("train", {})),
            sweep=SweepConfig(**data.get("sweep", {})),
            seed=int(data.get("seed", 0)),
            device=str(data.get("device", "cuda")),
            run_name=str(data.get("run_name", "phase1")),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
