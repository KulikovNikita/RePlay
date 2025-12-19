from contextlib import nullcontext as no_exception

import pytest
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from amazme.replay.models.nn.sequential.sampler import SequentialNegativeSampler

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize(
    "negative_sampling_strategy, sample_distribution, expected_exception",
    [
        ("custom_weights", None, pytest.raises(ValueError)),
        ("custom_weights", torch.FloatTensor([0.5, 2.0, 1.5, 0.78]), pytest.raises(ValueError)),
        ("custom_weights", torch.FloatTensor([0.5, 2.0, 1.5]), no_exception()),
        ("inbatch", torch.FloatTensor([0.5, 2.0, 1.5]), pytest.raises(ValueError)),
        ("wrong_strategy_name", None, pytest.raises(AssertionError)),
    ],
)
def test_negative_sampler_raises(negative_sampling_strategy, sample_distribution, expected_exception):
    with expected_exception:
        SequentialNegativeSampler(
            vocab_size=3,
            item_id_feature_name="item_id",
            negative_sampling_strategy=negative_sampling_strategy,
            num_negative_samples=2,
            sample_distribution=sample_distribution,
        )


@pytest.mark.torch
@pytest.mark.parametrize(
    "negative_sampling_strategy, sample_distribution, expected_output_shape",
    [
        ("inbatch", None, (4, 3)),
        ("global_uniform", None, (3,)),
        ("custom_weights", torch.FloatTensor([0.5, 2.0, 1.5]), (3,)),
    ],
)
def test_negatives_shape(simple_batch, negative_sampling_strategy, sample_distribution, expected_output_shape):
    sampler = SequentialNegativeSampler(
        vocab_size=3,
        item_id_feature_name="item_id",
        negative_sampling_strategy=negative_sampling_strategy,
        num_negative_samples=3,
        sample_distribution=sample_distribution,
    )
    negatives = sampler.get_negatives(simple_batch)

    assert negatives.size() == expected_output_shape
