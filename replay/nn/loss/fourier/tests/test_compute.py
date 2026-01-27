import torch
import pytest

from ..compute import compute_cdf_image

from .bin_distribution import (
    gen_bin_borders,
    make_full_rel_cdf,
    random_bin_distribution,
)


def trapezoid_var(xs: torch.Tensor, ys: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    ks = torch.arange(1, n_harmonics, dtype=torch.int64)
    x0, x1, y0, y1 = xs[:-1], xs[1:], ys[:-1], ys[1:]
    x_diff, y_diff = x1 - x0, y1 - y0
    denom = (torch.pi**2) * (ks**2)[None, :] * x_diff[:, None]
    variance = torch.cos(torch.pi * ks[None, :] * x0[:, None]) - torch.cos(
        torch.pi * ks[None, :] * x1[:, None]
    )
    var_term = y_diff[:, None] * variance / denom
    fixed = y1[:, None] * torch.sin(torch.pi * ks[None, :] * x1[:, None]) - y0[
        :, None
    ] * torch.sin(torch.pi * ks[None, :] * x0[:, None])
    fix_term = fixed / (torch.pi * ks[None, :])
    return fix_term - var_term


def trapezoid_const(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    x0, x1, y0, y1 = xs[:-1], xs[1:], ys[:-1], ys[1:]
    result = 0.25 * (x1 - x0) * (y1 + y0)
    return result[:, None]


def trapezoid(xs: torch.Tensor, ys: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    variable: torch.Tensor = trapezoid_var(xs, ys, n_harmonics)
    constant: torch.Tensor = trapezoid_const(xs, ys)
    return torch.hstack([constant, variable])


def trapezoid_full(
    xs: torch.Tensor, ys: torch.Tensor, n_harmonics: int
) -> torch.Tensor:
    return torch.sum(trapezoid(xs, ys, n_harmonics), dim=0)


def groundtruth_cdf_image(rel_profile: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    bin_count: int = torch.numel(rel_profile)
    rel_bin: torch.Tensor = gen_bin_borders(bin_count)
    rel_cdf: torch.Tensor = make_full_rel_cdf(rel_profile)
    all_cdfs: torch.Tensor = torch.cat([rel_cdf, rel_cdf[:-1].flip(-1)])
    all_bins: torch.Tensor = torch.cat([rel_bin, 2.0 - rel_bin[:-1].flip(-1)])
    return trapezoid_full(all_bins, all_cdfs, n_harmonics)


DEFAULT_ERROR: float = 1e-4
DEFAULT_SAMPLE_COUNT: int = int(1e5)

max_err_map: dict[int, float] = {
    16: 6e-4,
    128: 6e-4,
    1024: 6e-4,
}

mean_err_map: dict[int, float] = {
    16: 6e-4,
    128: 6e-4,
    1024: 6e-4,
}


@pytest.mark.parametrize("seed", [42, 777, 2007])
@pytest.mark.parametrize("bin_count", [4, 32, 64, 256])
@pytest.mark.parametrize("n_harmonics", [16, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compute(
    seed: int,
    bin_count: int,
    n_harmonics: int,
    dtype: torch.dtype,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> None:
    rel_profile: torch.Tensor
    samples: torch.Tensor
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    rel_profile, samples = random_bin_distribution(sample_count, bin_count, generator)
    rel_profile, samples = rel_profile.to(dtype=dtype), samples.to(dtype=dtype)

    gtr_image: torch.Tensor = groundtruth_cdf_image(
        rel_profile, n_harmonics=n_harmonics
    )
    cdf_image: torch.Tensor = compute_cdf_image(samples, n_harmonics=n_harmonics)

    abs_diff = torch.abs(cdf_image - gtr_image)
    max_abs_error = torch.max(abs_diff).cpu().item()
    max_error_boundary = max_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert max_abs_error <= max_error_boundary

    mean_abs_error = torch.mean(abs_diff).cpu().item()
    mean_error_boundary = mean_err_map.get(n_harmonics, DEFAULT_ERROR)
    assert mean_abs_error <= mean_error_boundary
