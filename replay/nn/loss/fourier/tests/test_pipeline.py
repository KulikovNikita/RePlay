import torch
import pytest

from ..evaluate import evaluate_image
from ..compute import compute_cdf_image

from .bin_distribution import make_cdf_interp, random_bin_distribution

DEFAULT_ERROR: float = 1e-4
DEFAULT_EVAL_COUNT: int = 65_536
DEFAULT_SAMPLE_COUNT: int = int(1e5)

max_err_map: dict[int, float] = {
    16: 3e-2,
    128: 5e-3,
    1024: 3e-3,
}

mean_err_map: dict[int, float] = {
    16: 6e-3,
    128: 6e-4,
    1024: 6e-4,
}


@pytest.mark.parametrize("seed", [42, 777, 2007])
@pytest.mark.parametrize("bin_count", [4, 32, 64, 256])
@pytest.mark.parametrize("n_harmonics", [16, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pipeline(
    seed: int,
    bin_count: int,
    n_harmonics: int,
    dtype: torch.dtype,
    eval_count: int = DEFAULT_EVAL_COUNT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> None:
    rel_profile: torch.Tensor
    samples: torch.Tensor
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    rel_profile, samples = random_bin_distribution(sample_count, bin_count, generator)
    rel_profile, samples = rel_profile.to(dtype=dtype), samples.to(dtype=dtype)

    eval_args: torch.Tensor = torch.linspace(0, 1, eval_count, dtype=dtype)
    cdf_image: torch.Tensor = compute_cdf_image(samples, n_harmonics)
    cdf_eval: torch.Tensor = evaluate_image(cdf_image, eval_args)
    cdf_interp: torch.Tensor = make_cdf_interp(rel_profile)(eval_args)

    abs_diff = torch.abs(cdf_eval - cdf_interp)
    max_abs_error = torch.max(abs_diff).cpu().item()
    max_error_boundary = max_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert max_abs_error <= max_error_boundary

    mean_abs_error = torch.mean(abs_diff).cpu().item()
    mean_error_boundary = mean_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert mean_abs_error <= mean_error_boundary
