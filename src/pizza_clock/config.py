from dataclasses import dataclass
import torch as t


@dataclass
class Config:
    p: int = 59
    attention_rate: float = 0.0
    residual_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 2.0
    wandb_project_name: str = "modular-addition"
    wandb_name: str | None = None
    use_wandb: bool = True
    device: str = "cpu"
    seed: int | None = 42


def get_device() -> str:
    if t.backends.mps.is_available():
        return "mps"
    elif t.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
