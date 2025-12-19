import copy

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    ParquetModule,
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
)
from replay.nn.transforms import (
    CopyTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
    UniformNegativeSamplingTransform,
    UnsqueezeTransform,
)


@pytest.fixture(scope="module")
def tensor_schema():
    tensor_schema = TensorSchema(
        [
            TensorFeatureInfo(
                name="item_id",
                is_seq=True,
                cardinality=41,
                padding_value=40,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                name="cat_list_feature",
                is_seq=True,
                cardinality=5,
                padding_value=4,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "cat_list_feature")],
            ),
            TensorFeatureInfo(
                name="num_feature",
                is_seq=True,
                tensor_dim=1,
                padding_value=0,
                embedding_dim=64,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_feature")],
            ),
            TensorFeatureInfo(
                name="num_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=6,
                embedding_dim=64,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_list_feature")],
            ),
        ]
    )
    return tensor_schema


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
    cat_list_feature_sequences = torch.LongTensor(
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
    # user_list_feature_sequences = torch.FloatTensor(  # rewrite to random
    #     [
    #         [
    #             [0.1167, 0.4971, 0.2617, 0.5642, 0.4445, 0.0070],
    #             [0.5000, 0.1506, 0.5602, 0.8270, 0.5071, 0.7313],
    #             [0.0930, 0.5950, 0.2436, 0.1521, 0.0081, 0.9209],
    #             [0.3076, 0.2529, 0.5184, 0.0434, 0.7885, 0.9906],
    #             [0.9827, 0.4891, 0.9374, 0.5577, 0.1775, 0.5455],
    #         ],
    #         [
    #             [0.3700, 0.2712, 0.5405, 0.2185, 0.0124, 0.7856],
    #             [0.2787, 0.5141, 0.4877, 0.9621, 0.7780, 0.8361],
    #             [0.6404, 0.7880, 0.8786, 0.8144, 0.3588, 0.2300],
    #             [0.8068, 0.7603, 0.1526, 0.6850, 0.3967, 0.7796],
    #             [0.5104, 0.1680, 0.0449, 0.2429, 0.6576, 0.0737],
    #         ],
    #         [
    #             [0.3935, 0.7009, 0.0801, 0.0288, 0.9336, 0.4312],
    #             [0.5848, 0.7728, 0.2201, 0.3426, 0.8623, 0.2485],
    #             [0.3278, 0.4886, 0.4630, 0.4874, 0.6532, 0.9900],
    #             [0.6753, 0.9985, 0.4158, 0.1668, 0.1952, 0.7187],
    #             [0.4071, 0.4536, 0.6778, 0.6317, 0.8205, 0.8902],
    #         ],
    #         [
    #             [0.8029, 0.4034, 0.9652, 0.5938, 0.1313, 0.8616],
    #             [0.7426, 0.8609, 0.2510, 0.0596, 0.9214, 0.0260],
    #             [0.8835, 0.9816, 0.8536, 0.8821, 0.8741, 0.2825],
    #             [0.9205, 0.2579, 0.8056, 0.3690, 0.4510, 0.3462],
    #             [0.4025, 0.2174, 0.9828, 0.0128, 0.4934, 0.6468],
    #         ],
    #     ]
    # )
    num_list_feature_sequences = torch.rand(4,5,8)
    
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
            "cat_list_feature": cat_list_feature_sequences,
            "num_feature": num_feature_sequences,
            "num_list_feature": num_list_feature_sequences,
        },
        "padding_mask": padding_mask,
    }

def generate_recsys_dataset(
    tensor_schema: TensorSchema,
    n_users: int = 10,
    max_len: int = 20,
    seed: int = 2,
    cat_list_item_size: int = 3,
    num_list_item_size: int = 5,
):
    np.random.seed(seed)

    rows = []
    for i in range(n_users):
        hist_len = np.random.randint(1, max_len + 1, size=None, dtype=int)
        row = {"user_id": i}

        for feature_info in tensor_schema.all_features:
            if not feature_info.is_seq:
                continue

            if feature_info.feature_type == FeatureType.CATEGORICAL:
                row[feature_info.name] = np.random.randint(0, feature_info.cardinality - 2, size=hist_len).tolist()
            elif feature_info.feature_type == FeatureType.CATEGORICAL_LIST:
                row[feature_info.name] = [
                    np.random.randint(0, feature_info.cardinality - 2, size=hist_len).tolist()
                    for _ in range(cat_list_item_size)
                ]
            elif feature_info.feature_type == FeatureType.NUMERICAL:
                row[feature_info.name] = np.random.random(size=(hist_len,)).tolist()
            elif feature_info.feature_type == FeatureType.NUMERICAL_LIST:
                row[feature_info.name] = [
                    np.random.random(size=(hist_len, feature_info.tensor_dim)).tolist()
                    for _ in range(num_list_item_size)
                ]

        rows.append(row)

    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def max_len():
    return 7

@pytest.fixture(scope="module")
def seed():
    return 1

@pytest.fixture(scope="module")
def parquet_module_path(tmp_path_factory, tensor_schema, seed, max_len):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / f"tmp_{seed}.parquet"

    df = generate_recsys_dataset(tensor_schema, n_users=50, max_len=max_len, seed=seed)
    df.to_parquet(path, index=False)

    return str(path)


@pytest.fixture(scope="module")
def parquet_module(parquet_module_path, tensor_schema, max_len, batch_size=16):
    transforms = {
        "train": [
            NextTokenTransform(
                label_field="item_id", query_features="user_id", shift=1, out_feature_name="positive_labels"
            ),
            RenameTransform(
                {"user_id": "query_id", "item_id_mask": "padding_mask", "labels_mask": "target_padding_mask"}
            ),
            UniformNegativeSamplingTransform(
                vocab_size=tensor_schema["item_id"].cardinality - 2, num_negative_samples=10
            ),
            UnsqueezeTransform("target_padding_mask", -1),
            UnsqueezeTransform("positive_labels", -1),
            GroupTransform({"features": tensor_schema.names}),
        ],
        "val": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            CopyTransform({"item_id": "train"}),
            CopyTransform({"item_id": "ground_truth"}),
            GroupTransform({"features": tensor_schema.names}),
        ],
        "test": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            GroupTransform({"features": tensor_schema.names}),
        ],
    }
    shared_meta = {"user_id": {}}
    shared_meta.update(
        {
            name: {"shape": max_len, "padding": feature_info.padding_value}
            for name, feature_info in tensor_schema.items()
        }
    )

    metadata = {
        "train": copy.deepcopy(shared_meta),
        "val": copy.deepcopy(shared_meta),
        "test": copy.deepcopy(shared_meta),
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        val_path=parquet_module_path,
        test_path=parquet_module_path,
    )
    return parquet_module


# @pytest.fixture(scope="module")
# def sequential_sample(parquet_module):
#     return parquet_module.compiled_transforms["train"](next(iter(parquet_module.train_dataloader())))
