import pytest


import torch

from replay.nn import InferenceOutput, TrainOutput


@pytest.mark.torch
def test_body_forward(model, simple_batch):
    output = model.body(simple_batch["feature_tensors"], simple_batch["padding_mask"])
    assert output.shape == (4, 7, 64)


# @pytest.mark.torch
# @pytest.mark.parametrize(
#     "wrong_sequential_sample",
#     [
#         pytest.param("missing field"),
#         pytest.param("wrong length"),
#         pytest.param("index out of embedding"),
#     ],
#     indirect=["wrong_sequential_sample"],
# )
# def test_wrong_input(model, wrong_sequential_sample):
#     with pytest.raises((AssertionError, IndexError, TypeError, KeyError, RuntimeError)):
#         model(**wrong_sequential_sample)


@pytest.mark.torch
def test_model_train_forward(model, simple_batch):
    model.train()
    output = model(**simple_batch)
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
def test_model_inference_forward(model, simple_batch, candidates_to_score, expected_shape):
    model.eval()
    output = model(simple_batch["feature_tensors"], simple_batch["padding_mask"], candidates_to_score)
    assert isinstance(output, InferenceOutput)
    assert output.logits.size() == expected_shape
    assert output.hidden_states[0].size() == (2, 7, 64)
