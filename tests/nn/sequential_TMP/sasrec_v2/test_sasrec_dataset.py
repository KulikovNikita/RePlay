import pandas as pd
import pytest
from replay.utils import TORCH_AVAILABLE

from amazme.replay.data import FeatureType

if TORCH_AVAILABLE:
    import torch
    from replay.data.nn.schema import TensorFeatureInfo, TensorSchema

    from amazme.replay.data.nn import PandasSequentialDataset
    from amazme.replay.models.nn.sequential.sasrec_v2 import SasRecTrainingDataset

torch = pytest.importorskip("torch")


@pytest.mark.torch
def test_sasrec_dataset_getitem(sasrec_training_dataset):
    sample = sasrec_training_dataset[0]

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
def test_sasrec_dataset_getitem_with_single_label(sequential_dataset):
    sasrec_training_dataset = SasRecTrainingDataset(sequential_dataset, max_sequence_length=1)
    sample = sasrec_training_dataset[0]

    assert sample["positive_labels"].dim() == 2
    assert sample["target_padding_mask"].dim() == 2


@pytest.mark.torch
def test_sasrec_dataset_add_negatives_in_collate(sasrec_training_dataset):
    collated_batch = sasrec_training_dataset.collate_fn([sasrec_training_dataset[0]])
    assert collated_batch["negative_labels"].numel() > 0


@pytest.mark.torch
def test_wrong_label_feature_name(sequential_dataset):
    with pytest.raises(ValueError):
        SasRecTrainingDataset(
            sequential_dataset, label_feature_name="incorrect_label_feature_name", max_sequence_length=7
        )


@pytest.mark.torch
@pytest.mark.parametrize(
    "incorrect_schema",
    [
        TensorSchema(
            [
                TensorFeatureInfo(name="some_label_feature", is_seq=True, feature_type=FeatureType.NUMERICAL),
            ]
        ),
        TensorSchema(
            [
                TensorFeatureInfo(name="some_label_feature", is_seq=False, feature_type=FeatureType.CATEGORICAL),
            ]
        ),
    ],
    ids=["not categorical", "not sequential"],
)
def test_wrong_label_feature_type(incorrect_schema):
    sequences = pd.DataFrame(
        {
            "user_id": [0, 1],
            "some_label_feature": [
                [0, 1, 2],
                [0, 0, 1, 2],
            ],
        }
    )
    sequential_dataset = PandasSequentialDataset(
        tensor_schema=incorrect_schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )
    with pytest.raises(ValueError):
        SasRecTrainingDataset(sequential_dataset, label_feature_name=incorrect_schema.names[0], max_sequence_length=7)
