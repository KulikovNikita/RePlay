import numpy as np
import pandas as pd
import pytest

from amazme.replay.data import FeatureHint, FeatureSource, FeatureType
from amazme.replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from replay.data.nn.schema import TensorFeatureInfo, TensorSchema

    from amazme.replay.data.nn import PandasSequentialDataset, TensorFeatureSource
    from amazme.replay.experimental.nn.data.schema_builder import TensorSchemaBuilder
    from amazme.replay.models.nn.sequential.bert4rec import Bert4RecTrainingDataset, Bert4RecValidationDataset
    from amazme.replay.models.nn.sequential.sasrec_v2 import SasRecPredictionDataset, SasRecValidationDataset

torch = pytest.importorskip("torch")


@pytest.fixture(scope="package")
def item_user_sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, np.array([0, 1, 1, 1, 2])),
            (1, np.array([0, 1, 3, 1, 2])),
            (2, np.array([0, 2, 3, 1, 2])),
            (3, np.array([1, 2, 0, 1, 2])),
        ],
        columns=[
            "user_id",
            "item_id",
        ],
    )

    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_hint=FeatureHint.ITEM_ID,
        )
        .build()
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="module")
def train_loader(item_user_sequential_dataset):
    train = Bert4RecTrainingDataset(item_user_sequential_dataset, 5)
    return torch.utils.data.DataLoader(train)


@pytest.fixture(scope="module")
def val_loader(item_user_sequential_dataset):
    val = Bert4RecValidationDataset(
        item_user_sequential_dataset, item_user_sequential_dataset, item_user_sequential_dataset, max_sequence_length=5
    )
    return torch.utils.data.DataLoader(val)


@pytest.fixture(scope="module")
def tensor_schema():
    tensor_schema = TensorSchema(
        [
            TensorFeatureInfo(
                name="item_id",
                is_seq=True,
                cardinality=3,
                padding_value=3,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                name="item_list_feature",
                is_seq=True,
                cardinality=4,
                padding_value=4,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_list_feature")],
            ),
            TensorFeatureInfo(
                name="num_feature",
                is_seq=True,
                tensor_dim=1,
                padding_value=0,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "num_feature")],
            ),
            TensorFeatureInfo(
                name="user_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=6,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_list_feature")],
            ),
        ]
    )
    return tensor_schema


@pytest.fixture(scope="module")
def sequential_dataset(tensor_schema):
    sequences = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "item_id": [
                [1, 2],
                [1],
                [2, 0, 2, 1, 2],
                [2, 1, 0],
            ],
            "item_list_feature": [
                [[3, 2, 1], [2, 4, 3]],
                [[3, 2, 1]],
                [[2, 4, 3], [4, 0, 3], [2, 4, 3], [3, 2, 1], [2, 4, 3]],
                [[2, 4, 3], [3, 2, 1], [4, 0, 3]],
            ],
            "num_feature": [
                [0.0, 1.0, 2.0],
                [1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 2.0, 2.0],
            ],
            "user_list_feature": [
                [
                    [0.1167, 0.4971, 0.2617, 0.5642, 0.4445, 0.0070],
                    [0.5000, 0.1506, 0.5602, 0.8270, 0.5071, 0.7313],
                    [0.0930, 0.5950, 0.2436, 0.1521, 0.0081, 0.9209],
                    [0.3076, 0.2529, 0.5184, 0.0434, 0.7885, 0.9906],
                    [0.9827, 0.4891, 0.9374, 0.5577, 0.1775, 0.5455],
                ],
                [
                    [0.3700, 0.2712, 0.5405, 0.2185, 0.0124, 0.7856],
                    [0.2787, 0.5141, 0.4877, 0.9621, 0.7780, 0.8361],
                    [0.6404, 0.7880, 0.8786, 0.8144, 0.3588, 0.2300],
                    [0.8068, 0.7603, 0.1526, 0.6850, 0.3967, 0.7796],
                    [0.5104, 0.1680, 0.0449, 0.2429, 0.6576, 0.0737],
                ],
                [
                    [0.3935, 0.7009, 0.0801, 0.0288, 0.9336, 0.4312],
                    [0.5848, 0.7728, 0.2201, 0.3426, 0.8623, 0.2485],
                    [0.3278, 0.4886, 0.4630, 0.4874, 0.6532, 0.9900],
                    [0.6753, 0.9985, 0.4158, 0.1668, 0.1952, 0.7187],
                    [0.4071, 0.4536, 0.6778, 0.6317, 0.8205, 0.8902],
                ],
                [
                    [0.8029, 0.4034, 0.9652, 0.5938, 0.1313, 0.8616],
                    [0.7426, 0.8609, 0.2510, 0.0596, 0.9214, 0.0260],
                    [0.8835, 0.9816, 0.8536, 0.8821, 0.8741, 0.2825],
                    [0.9205, 0.2579, 0.8056, 0.3690, 0.4510, 0.3462],
                    [0.4025, 0.2174, 0.9828, 0.0128, 0.4934, 0.6468],
                ],
            ],
        }
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=tensor_schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )
    return sequential_dataset


@pytest.fixture(scope="module")
def simple_batch():
    item_sequences = torch.LongTensor(
        [
            [3, 3, 3, 1, 2],
            [3, 0, 0, 1, 2],
            [2, 0, 2, 1, 2],
            [3, 3, 2, 1, 0],
        ],
    )
    item_list_feature_sequences = torch.LongTensor(
        [
            [[4, 4, 4], [4, 4, 4], [4, 4, 4], [3, 2, 1], [2, 4, 3]],
            [[4, 4, 4], [4, 0, 3], [4, 0, 3], [3, 2, 1], [2, 4, 3]],
            [[2, 4, 3], [4, 0, 3], [2, 4, 3], [3, 2, 1], [2, 4, 3]],
            [[4, 4, 4], [4, 4, 4], [2, 4, 3], [3, 2, 1], [4, 0, 3]],
        ]
    )
    num_feature_sequences = torch.FloatTensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0], [0, 0.0, 1.0, 1.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 2.0, 2.0, 2.0]]
    )
    user_list_feature_sequences = torch.FloatTensor(
        [
            [
                [0.1167, 0.4971, 0.2617, 0.5642, 0.4445, 0.0070],
                [0.5000, 0.1506, 0.5602, 0.8270, 0.5071, 0.7313],
                [0.0930, 0.5950, 0.2436, 0.1521, 0.0081, 0.9209],
                [0.3076, 0.2529, 0.5184, 0.0434, 0.7885, 0.9906],
                [0.9827, 0.4891, 0.9374, 0.5577, 0.1775, 0.5455],
            ],
            [
                [0.3700, 0.2712, 0.5405, 0.2185, 0.0124, 0.7856],
                [0.2787, 0.5141, 0.4877, 0.9621, 0.7780, 0.8361],
                [0.6404, 0.7880, 0.8786, 0.8144, 0.3588, 0.2300],
                [0.8068, 0.7603, 0.1526, 0.6850, 0.3967, 0.7796],
                [0.5104, 0.1680, 0.0449, 0.2429, 0.6576, 0.0737],
            ],
            [
                [0.3935, 0.7009, 0.0801, 0.0288, 0.9336, 0.4312],
                [0.5848, 0.7728, 0.2201, 0.3426, 0.8623, 0.2485],
                [0.3278, 0.4886, 0.4630, 0.4874, 0.6532, 0.9900],
                [0.6753, 0.9985, 0.4158, 0.1668, 0.1952, 0.7187],
                [0.4071, 0.4536, 0.6778, 0.6317, 0.8205, 0.8902],
            ],
            [
                [0.8029, 0.4034, 0.9652, 0.5938, 0.1313, 0.8616],
                [0.7426, 0.8609, 0.2510, 0.0596, 0.9214, 0.0260],
                [0.8835, 0.9816, 0.8536, 0.8821, 0.8741, 0.2825],
                [0.9205, 0.2579, 0.8056, 0.3690, 0.4510, 0.3462],
                [0.4025, 0.2174, 0.9828, 0.0128, 0.4934, 0.6468],
            ],
        ]
    )
    padding_mask = torch.BoolTensor(
        [
            [0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
    )

    return {
        "feature_tensors": {
            "item_id": item_sequences,
            "item_list_feature": item_list_feature_sequences,
            "num_feature": num_feature_sequences,
            "user_list_feature": user_list_feature_sequences,
        },
        "padding_mask": padding_mask,
    }


@pytest.fixture(scope="module")
def validation_dataloader(sequential_dataset):
    validation_dataset = SasRecValidationDataset(
        sequential_dataset, sequential_dataset, sequential_dataset, max_sequence_length=7
    )
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1)
    return validation_dataloader


@pytest.fixture(scope="module")
def test_dataloader(sequential_dataset):
    pred_dataset = SasRecPredictionDataset(sequential_dataset, max_sequence_length=7)
    test_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=1)
    return test_dataloader
