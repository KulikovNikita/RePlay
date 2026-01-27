import torch

import typing as tp

def prepare_x_mask(x: torch.Tensor, mask: torch.Tensor | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
    if mask is not None:
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(x).flatten()
    x = x.flatten()

    return (x, mask)


def get_n_elems(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        result: torch.Tensor = torch.tensor(torch.numel(x), device=x.device)
    else:
        casted: torch.Tensor = tp.cast(torch.Tensor, mask)
        result: torch.Tensor = torch.sum(casted, dtype=torch.int64)

    assert result.dtype == torch.int64
    assert result.device == x.device

    return result


def compute_cdf_cn(
    x: torch.Tensor, n_harmonics: int, mask: torch.Tensor | None = None
) -> torch.Tensor:
    ks: torch.Tensor = torch.arange(1, n_harmonics, dtype=x.dtype, device=x.device)
    k_coef: torch.Tensor = (2.0 * (-1) ** ks / (torch.pi * ks)).to(dtype=x.dtype)
    sin: torch.Tensor = torch.sin(torch.pi * ks[None, :] * (1.0 - x[:, None]))
    body: torch.Tensor = k_coef[None, :] * sin
    if mask is not None:
        casted: torch.Tensor = tp.cast(torch.Tensor, mask)
        body = body * casted[:, None]
    return torch.sum(body, dim=0)


def compute_cdf_c0(x: torch.Tensor, mask: torch.Tensor | None = None,
                   n_elems: torch.Tensor | None = None,) -> torch.Tensor:
    if n_elems is None:
        n_elems = get_n_elems(x, mask)
    casted_n_elems: torch.Tensor = tp.cast(torch.Tensor, n_elems)

    if mask is None:
        sum: torch.Tensor = torch.sum(x)
    else:
        casted: torch.Tensor = tp.cast(torch.Tensor, mask)
        sum: torch.Tensor = torch.sum(x * casted)

    return casted_n_elems - sum

def compute_cdf_image(
    x: torch.Tensor, n_harmonics: int, 
    mask: torch.Tensor | None = None,
    n_elems: torch.Tensor | None = None,
) -> torch.Tensor:
    x, mask = prepare_x_mask(x, mask)

    if n_elems is None:
        n_elems = get_n_elems(x, mask)
    casted_n_elems: torch.Tensor = tp.cast(torch.Tensor, n_elems)

    c0: torch.Tensor = compute_cdf_c0(x, mask, n_elems)
    cn: torch.Tensor = compute_cdf_cn(x, n_harmonics, mask)

    result = torch.cat([c0[None], cn]) / casted_n_elems

    assert result.requires_grad == x.requires_grad
    assert torch.numel(result) == n_harmonics
    assert result.device == x.device

    return result
