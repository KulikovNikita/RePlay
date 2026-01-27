import torch

import torch.nn.functional as func

import typing as tp

type NormType = tp.Literal["none", "l2"]


def normalize(x: torch.Tensor, dim: int = -1, norm: NormType = "l2") -> torch.Tensor:
    if norm == "none":
        result: torch.Tensor = x
    elif norm == "l2":
        result: torch.Tensor = func.normalize(x, dim=dim)
    else:
        msg: str = f"Not supported norm: {norm=}"
        raise ValueError(msg)
    return result
