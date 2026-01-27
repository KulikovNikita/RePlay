import torch

import pytest

from ..compute import compute_cdf_image

@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("n_harmonics", [16, 64, 512])
@pytest.mark.parametrize("n_pos", [int(1e3), int(1e5)])
@pytest.mark.parametrize("n_neg", [int(1e3), int(1e5)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_masked(seed: int, n_harmonics: int, n_pos: int, n_neg: int, dtype: torch.dtype) -> None:
    gen = torch.Generator().manual_seed(seed)
    
    pos_mean = torch.tensor(0.4).expand(n_pos)
    pos_samples = torch.normal(pos_mean, 0.3, generator = gen)
    pos_samples = torch.clamp(pos_samples, 0.0, 1.0)

    neg_mean = torch.tensor(0.6).expand(n_neg)
    neg_samples = torch.normal(neg_mean, 0.3, generator = gen)
    neg_samples = torch.clamp(neg_samples, 0.0, 1.0)

    perm = torch.randperm(n_pos + n_neg, generator = gen)
    all_samples = torch.cat((pos_samples, neg_samples))[perm]
    is_pos = (perm < n_pos).to(dtype = dtype)

    pure_pos = compute_cdf_image(pos_samples, n_harmonics)
    mixed = compute_cdf_image(all_samples, n_harmonics, is_pos)

    assert torch.allclose(pure_pos, mixed)
