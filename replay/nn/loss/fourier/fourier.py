import torch
import warnings

from typing import Self, Callable, Optional, cast

from .normalize import normalize, NormType
from .evaluate import evaluate_image
from .compute import compute_cdf_image, prepare_x_mask, get_n_elems

DEFAULT_BATCH_SIZE: int = 1024


class InBatchFourierLoss(torch.nn.Module):
    def __init__(
        self: Self,
        n_harmonics: int,
        norm: NormType = "l2",
        sampling_size: int | float | None = None,
        image_momentum: float | None = None,
        batch_size: int | None = None,
        scaler: Callable | None = None,
    ) -> None:
        super().__init__()

        self.norm: NormType = norm
        self.n_harmonics: int = n_harmonics

        self.scaler: Callable | None = scaler

        self.sampling_size: int | float | None = sampling_size
        self.image_momentum: float | None = image_momentum

        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        self.batch_size: int = cast(int, batch_size)

        self.has_image: bool = False
        self.cdf_image: torch.nn.Buffer = torch.nn.Buffer(
            data=torch.zeros(
                size=(self.n_harmonics,),
                dtype=torch.float32,
            ),
            persistent=True,
        )

    @property
    def item_embeddings_callback(
        self,
    ) -> Callable[[Optional[torch.Tensor]], torch.Tensor]:
        if self._item_embeddings_callback is None:
            msg = "The callback for getting item embeddings is not defined"
            raise AttributeError(msg)
        return self._item_embeddings_callback

    @item_embeddings_callback.setter
    def item_embeddings_callback(self, func: Optional[Callable]) -> None:
        self._item_embeddings_callback = func

    def normalize(self: Self, embeddings: torch.Tensor) -> torch.Tensor:
        return normalize(embeddings, norm=self.norm)

    def get_sampling_size(self: Self, n_embeddings: int) -> int:
        match self.sampling_size:
            case None:
                raw_size = n_embeddings
            case int():
                raw_size = self.sampling_size
            case float():
                raw_size = n_embeddings * self.sampling_size
            case _:
                msg: str = f"Unsupported type: {self.sampling_size=}."
                raise ValueError(msg)
        if n_embeddings < raw_size:
            msg: str = f"Suspicious: {raw_size=} vs. {n_embeddings=}."
            warnings.warn(msg)
        return int(raw_size)

    def get_image_momentum(self: Self, n_embeddings: int, sampling_size: int) -> float:
        if self.image_momentum is None:
            raw_momentum = sampling_size / n_embeddings
        else:
            raw_momentum = float(self.image_momentum)
        if not self.has_image:
            raw_momentum = 1.0
        if 1.0 < raw_momentum:
            msg: str = f"Momentum will be capped: {raw_momentum=}"
            warnings.warn(msg)
        return float(raw_momentum)

    def norm_scores(self: Self, scores: torch.Tensor) -> torch.Tensor:
        return 0.5 * (scores + 1.0)

    def get_samples(
        self: Self, item_embeddings: torch.Tensor, sampling_size: int
    ) -> torch.Tensor:
        n_embeddings: int = item_embeddings.size(0)
        max_raw_id: int = max(n_embeddings, sampling_size)
        raw_ids: torch.Tensor = torch.randperm(max_raw_id)
        result: torch.Tensor = raw_ids[:sampling_size] % n_embeddings
        return result.to(device=item_embeddings.device)

    def compute_cdf_image(
        self: Self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        n_elems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_cdf_image(x, self.n_harmonics, mask, n_elems)

    def compute_new_image(
        self: Self,
        embeddings: torch.Tensor,
        samples: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batches: tuple[torch.Tensor, ...] = torch.split(samples, self.batch_size)

        norm_embedding: torch.Tensor = self.normalize(embeddings)

        images, n_elems = 0.0, 0.0
        for batch in batches:
            part_item_embeddings: torch.Tensor = self.item_embeddings_callback(batch)
            part_item_embeddings = self.normalize(part_item_embeddings)

            scores: torch.Tensor = torch.einsum(
                "...h,ih->...i", norm_embedding, part_item_embeddings
            )

            scores = self.norm_scores(scores)

            scores, casted_mask = prepare_x_mask(scores, mask)
            curr_n_elems: torch.Tensor = get_n_elems(scores, casted_mask)
            curr_image: torch.Tensor = self.compute_cdf_image(
                scores, casted_mask, curr_n_elems
            )

            images = curr_image * curr_n_elems + images
            n_elems = curr_n_elems + n_elems

        return cast(torch.Tensor, (images / n_elems))

    def get_cdf_image(
        self: Self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_item_embeddings = self.item_embeddings_callback(None)
        item_embeddings: torch.Tensor = cast(torch.Tensor, raw_item_embeddings)

        n_embeddings: int = item_embeddings.size(0)
        sampling_size: int = self.get_sampling_size(n_embeddings)
        image_momentum: float = self.get_image_momentum(n_embeddings, sampling_size)

        samples: torch.Tensor = self.get_samples(item_embeddings, sampling_size)

        new_image: torch.Tensor = self.compute_new_image(embeddings, samples, mask)

        result_image: torch.Tensor = (
            image_momentum * new_image + (1.0 - image_momentum) * self.cdf_image
        )

        self.cdf_image.copy_(result_image.detach())

        self.has_image = True
        return result_image

    def get_raw_scores(
        self: Self, embeddings: torch.Tensor, positive_labels: torch.Tensor
    ) -> torch.Tensor:
        raw_gtr_embeds: torch.Tensor = self.item_embeddings_callback(positive_labels)
        gtr_embeds: torch.Tensor = self.normalize(raw_gtr_embeds.squeeze())
        norm_embeds: torch.Tensor = self.normalize(embeddings)
        scores: torch.Tensor = torch.sum(gtr_embeds * norm_embeds, dim=-1)
        return self.norm_scores(scores)

    def forward(
        self: Self,
        model_embeddings: torch.Tensor,
        feature_tensors: "TensorMap",  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        image: torch.Tensor = self.get_cdf_image(model_embeddings, target_padding_mask)
        pos_scores: torch.Tensor = self.get_raw_scores(
            model_embeddings, positive_labels
        )
        raw_loss: torch.Tensor = 1.0 - evaluate_image(image, pos_scores)
        raw_loss = torch.sum(raw_loss * target_padding_mask.squeeze())
        loss_norm: torch.Tensor = torch.sum(target_padding_mask)
        return raw_loss / loss_norm
