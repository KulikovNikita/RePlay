import torch
import pytest

from ..evaluate import evaluate_image

from .bin_distribution import (
    gen_rel_profile,
    make_cdf_interp,
)

from .test_compute import (
    groundtruth_cdf_image,
)

DEFAULT_ERROR: float = 1e-4
DEFAULT_EVAL_COUNT: int = int(1e5)

max_err_map: dict[int, float] = {
    16: 3e-2,
    128: 4e-3,
    1024: 5e-4,
}

mean_err_map: dict[int, float] = {
    16: 6e-3,
    128: 6e-4,
    1024: 2e-5,
}


@pytest.mark.parametrize("seed", [42, 777, 2007])
@pytest.mark.parametrize("bin_count", [4, 32, 64, 256])
@pytest.mark.parametrize("n_harmonics", [16, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_evaluate(
    seed: int,
    bin_count: int,
    n_harmonics: int,
    dtype: torch.dtype,
    eval_count: int = DEFAULT_EVAL_COUNT,
) -> None:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    rel_profile: torch.Tensor = gen_rel_profile(bin_count, generator)

    xs: torch.Tensor = torch.linspace(0, 1, eval_count, dtype=dtype)
    cdf_image: torch.Tensor = groundtruth_cdf_image(rel_profile, n_harmonics)
    cdf_eval: torch.Tensor = evaluate_image(cdf_image, xs)
    cdf_interp: torch.Tensor = make_cdf_interp(rel_profile)(xs)

    abs_diff = torch.abs(cdf_eval - cdf_interp)
    max_abs_error = torch.max(abs_diff).cpu().item()
    max_error_boundary = max_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert max_abs_error <= max_error_boundary

    mean_abs_error = torch.mean(abs_diff).cpu().item()
    mean_error_boundary = mean_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert mean_abs_error <= mean_error_boundary
