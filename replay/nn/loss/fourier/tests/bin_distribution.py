import torch

from typing import Callable, List, Tuple


def gen_rel_profile(bin_count: int, generator: torch.Generator) -> torch.Tensor:
    rel_profile: torch.Tensor = torch.rand(
        bin_count, generator=generator, dtype=torch.float64
    )
    rel_profile = rel_profile / torch.sum(rel_profile)
    return rel_profile


def rel_to_abs_profile(sample_count: int, rel_profile: torch.Tensor) -> torch.Tensor:
    abs_profile: torch.Tensor = torch.floor(rel_profile * sample_count)
    abs_profile = torch.clip(abs_profile, min=0, max=sample_count)
    return abs_profile.to(dtype=torch.int64)


def gen_bin_borders(bin_count: int) -> torch.Tensor:
    raw: torch.Tensor = torch.arange(bin_count + 1, dtype=torch.float64)
    return raw * (1.0 / bin_count)


def gen_binned_samples(
    abs_profile: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    cpu_abs_profile: torch.Tensor = abs_profile.cpu()
    bin_count: int = torch.numel(cpu_abs_profile)

    cpu_bin_borders: torch.Tensor = gen_bin_borders(bin_count).cpu()
    bin_slices: List[torch.Tensor] = list()

    bin: int
    for bin in range(bin_count):
        count: int = cpu_abs_profile[bin].item()

        if 0 < count:
            lower: float = cpu_bin_borders[bin].item()
            upper: float = cpu_bin_borders[bin + 1].item()
            raw_slice: torch.Tensor = torch.rand(
                count, generator=generator, dtype=torch.float64
            )
            bin_slice: torch.Tensor = lower + (upper - lower) * raw_slice
            bin_slices.append(bin_slice)

    samples: torch.Tensor = torch.cat(bin_slices).to(dtype=torch.float64)
    assert torch.numel(samples) == torch.sum(cpu_abs_profile).item()
    return samples


def permute_samples(samples: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    perm: torch.LongTensor = torch.randperm(torch.numel(samples), generator=generator)
    return torch.take(samples, perm)


def random_bin_distribution(
    sample_count: int, bin_count: int, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    rel_profile: torch.Tensor = gen_rel_profile(bin_count, generator)
    abs_profile: torch.Tensor = rel_to_abs_profile(sample_count, rel_profile)

    binned_samples: torch.Tensor = gen_binned_samples(abs_profile, generator)
    perm_samples: torch.Tensor = permute_samples(binned_samples, generator)
    return (rel_profile, perm_samples)


def make_full_rel_cdf(rel_profile: torch.Tensor) -> torch.Tensor:
    zero: torch.Tensor = torch.asarray([0], dtype=rel_profile.dtype)
    return torch.cumsum(torch.cat([zero, rel_profile]), dim=-1)


def make_cdf_interp(
    rel_profile: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    bin_count: int = torch.numel(rel_profile)
    borders: torch.Tensor = gen_bin_borders(bin_count)
    full_rel_cdf: torch.Tensor = make_full_rel_cdf(rel_profile)

    def interp(x: torch.Tensor) -> torch.Tensor:
        bins = torch.searchsorted(borders, x, side="right")
        bins = torch.clip(bins, min=0, max=bin_count)

        left_bins: torch.Tensor
        right_bins: torch.Tensor
        left_cdfs: torch.Tensor
        right_cdfs: torch.Tensor
        left_bins, right_bins = borders[bins - 1], borders[bins]
        left_cdfs, right_cdfs = full_rel_cdf[bins - 1], full_rel_cdf[bins]

        rel_positions: torch.Tensor = (x - left_bins) / (right_bins - left_bins)
        result: torch.Tensor = (
            1.0 - rel_positions
        ) * left_cdfs + rel_positions * right_cdfs
        return torch.clip(result, min=0.0, max=1.0).to(dtype=x.dtype)

    return interp
