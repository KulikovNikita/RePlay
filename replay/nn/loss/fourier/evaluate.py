import torch


def evaluate_image(image: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    n_harmonics: int = image.size(-1)

    ks: torch.Tensor = torch.arange(n_harmonics, device=x.device, dtype=x.dtype)

    while image.ndim <= x.ndim:
        image = image.unsqueeze(0)

    while ks.ndim <= x.ndim:
        ks = ks.unsqueeze(0)

    x = x.unsqueeze(-1)

    body: torch.Tensor = image * torch.cos(torch.pi * ks * x)
    result: torch.Tensor = torch.sum(body, dim=-1)
    assert torch.numel(result) == torch.numel(x)

    return result
