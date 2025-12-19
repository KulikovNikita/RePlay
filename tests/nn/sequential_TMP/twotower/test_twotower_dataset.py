import pytest
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

torch = pytest.importorskip("torch")


@pytest.mark.torch
def test_twotower_dataset_getitem(twotower_training_dataset):
    sample = twotower_training_dataset[0]

    assert "feature_tensors" in sample
    assert "padding_mask" in sample
    assert "positive_labels" in sample
    assert "negative_labels" in sample
    assert "target_padding_mask" in sample

    torch.testing.assert_close(sample["feature_tensors"]["item_id"], torch.LongTensor([3, 3, 3, 3, 3, 3, 1]))
    torch.testing.assert_close(sample["padding_mask"], torch.BoolTensor([0, 0, 0, 0, 0, 0, 1]))
    torch.testing.assert_close(sample["positive_labels"], torch.LongTensor([[3], [3], [3], [3], [3], [1], [2]]))
    torch.testing.assert_close(sample["target_padding_mask"], torch.BoolTensor([[0], [0], [0], [0], [0], [1], [1]]))
    assert sample["negative_labels"].numel() == 0


@pytest.mark.torch
def test_twotower_dataset_add_negatives_in_collate(twotower_training_dataset):
    collated_batch = twotower_training_dataset.collate_fn([twotower_training_dataset[0]])
    assert collated_batch["negative_labels"].numel() > 0
