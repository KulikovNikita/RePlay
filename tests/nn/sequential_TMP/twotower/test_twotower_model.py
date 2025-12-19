import pytest

from amazme.replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from amazme.replay.data.nn import TensorMap
    from amazme.replay.models.nn.output import InferenceOutput, TrainOutput
    from amazme.replay.models.nn.sequential.twotower import TwoTowerBuilder


torch = pytest.importorskip("torch")


@pytest.mark.torch
def test_query_tower_forward(model, sequential_sample):
    output = model.body.query_tower(sequential_sample["feature_tensors"], sequential_sample["padding_mask"])
    assert output.shape == (2, 7, 64)


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score, expected_shape",
    [
        (torch.LongTensor([1]), (1, 64)),
        (torch.LongTensor([0, 1, 2]), (3, 64)),
        (None, (3, 64)),
    ],
)
def test_item_tower_forward(model, candidates_to_score, expected_shape):
    output = model.body.item_tower(candidates_to_score)
    assert output.shape == expected_shape


@pytest.mark.torch
@pytest.mark.parametrize(
    "incorrect_sequential_sample",
    [
        pytest.param("missing field"),
        pytest.param("wrong length"),
        pytest.param("index out of embedding"),
    ],
    indirect=["incorrect_sequential_sample"],
)
def test_incorrect_input(model, incorrect_sequential_sample):
    with pytest.raises((AssertionError, IndexError, TypeError, KeyError, RuntimeError)):
        model(**incorrect_sequential_sample)


@pytest.mark.torch
def test_incorrect_build_model(builder):
    builder = builder.loss(None)
    with pytest.raises(ValueError):
        builder.build()


class DummyContextMerger(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_hidden_state: torch.Tensor, feature_tensors: TensorMap) -> torch.Tensor:
        return model_hidden_state


@pytest.mark.torch
def test_model_forward_with_dummy_context_merger(builder, sequential_sample):
    model = builder.context_merger(DummyContextMerger()).build()
    model(**sequential_sample)


@pytest.mark.torch
def test_model_with_exclude_all_features(tensor_schema, dummy_mapping_path, dummy_reference_path):
    query_tower_names = ["item_id"]
    item_tower_names = ["item_list_feature"]
    builder = (
        TwoTowerBuilder()
        .ecom(
            tensor_schema=tensor_schema,
            feature_mapping_path=dummy_mapping_path,
            item_reference_path=dummy_reference_path,
            excluded_features=query_tower_names,
        )
        .query_tower_feature_names(query_tower_names)
        .item_tower_feature_names(item_tower_names)
    )
    with pytest.raises(ValueError):
        builder.build()


@pytest.mark.torch
def test_model_train_forward(model, sequential_sample):
    model.train()
    output = model(**sequential_sample)
    assert isinstance(output, TrainOutput)
    assert output.loss.ndim == 0
    assert output.hidden_states[0].size() == (2, 7, 64)


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score, expected_shape",
    [
        (torch.LongTensor([1]), (2, 1)),
        (torch.LongTensor([0, 1, 2]), (2, 3)),
        (None, (2, 3)),
    ],
)
def test_model_inference_forward(model, sequential_sample, candidates_to_score, expected_shape):
    model.eval()
    output = model(sequential_sample["feature_tensors"], sequential_sample["padding_mask"], candidates_to_score)
    assert isinstance(output, InferenceOutput)
    assert output.logits.size() == expected_shape
    assert output.hidden_states[0].size() == (2, 7, 64)
