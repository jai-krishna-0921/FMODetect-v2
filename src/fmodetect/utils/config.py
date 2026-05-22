"""YAML config loader -> nested dotdict via pydantic."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataCfg(BaseModel):
    h5_path: str
    val_fraction: float = 0.04
    num_workers: int = 2
    pin_memory: bool = True


class ModelCfg(BaseModel):
    in_channels: int = 6
    base_channels: list[int] = Field(default_factory=lambda: [16, 64, 128, 256, 256])
    use_cbam: bool = True
    predict_matting: bool = True
    predict_uncertainty: bool = True


class LossCfg(BaseModel):
    matting_weight: float = 0.5
    boundary_weight: float = 0.1


class TrainCfg(BaseModel):
    epochs: int = 60
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 1e-5
    amp: bool = True
    channels_last: bool = True
    grad_clip_norm: float = 1.0
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    early_stop_patience: int = 12


class LoggingCfg(BaseModel):
    mlflow_uri: str = "file:./experiments/mlruns"
    mlflow_experiment: str = "fmodetect-v2"
    use_clearml: bool = False
    clearml_project: str = "fmodetect-v2"
    tensorboard_dir: str = "experiments/tb"


class CheckpointCfg(BaseModel):
    dir: str = "experiments/checkpoints"
    keep_best_n: int = 3


class Config(BaseModel):
    seed: int = 42
    device: str = "cuda"
    data: DataCfg
    model: ModelCfg = ModelCfg()
    loss: LossCfg = LossCfg()
    train: TrainCfg = TrainCfg()
    logging: LoggingCfg = LoggingCfg()
    checkpoints: CheckpointCfg = CheckpointCfg()


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
