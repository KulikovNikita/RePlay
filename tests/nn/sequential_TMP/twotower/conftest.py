import copy
import pickle

import pandas as pd
import pytest

from amazme.replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from amazme.replay.models.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
    from amazme.replay.models.nn.sequential.common.agg import ConcatAggregator, SumAggregator
    from amazme.replay.models.nn.sequential.common.diff_attention import DiffTransformerBlock
    from amazme.replay.models.nn.sequential.common.embedding import SequentialEmbedder
    from amazme.replay.models.nn.sequential.common.ffn import SwiGLUEncoder
    from amazme.replay.models.nn.sequential.common.mask import DefaultAttentionMaskBuilder
    from amazme.replay.models.nn.sequential.common.normalization import RMSNorm
    from amazme.replay.models.nn.sequential.sampler import SequentialNegativeSampler
    from amazme.replay.models.nn.sequential.twotower import (
        QueryTowerEmbeddingAggregator,
        TwoTowerBuilder,
        TwoTowerTrainingDataset,
    )


@pytest.fixture(scope="module")
def dummy_mapping_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("twotower")
    path = tmp_dir / "dummy_mapping.pickle"

    mapping = {"item_id": {0: 0, 10: 1, 20: 2}, "item_list_feature": {1: 0, 11: 1, 21: 2, 31: 3, 41: 4}}
    with open(path, "wb") as f:
        pickle.dump(mapping, f)

    return str(path)


@pytest.fixture(scope="module")
def dummy_reference_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("twotower")
    path = tmp_dir / "dummy_reference.parquet"

    df = pd.DataFrame(data={"item_id": [0, 10, 20], "item_list_feature": [[41, 1, 31], [31, 21, 11], [21, 41, 31]]})
    df.to_parquet(path)

    return str(path)


@pytest.fixture(scope="module")
def twotower_training_dataset(sequential_dataset, tensor_schema, request):
    use_negative_sampling, negative_sampling_strategy = getattr(request, "param", (True, "global_uniform"))

    if use_negative_sampling:
        negative_sampler = SequentialNegativeSampler(
            vocab_size=tensor_schema["item_id"].cardinality,
            item_id_feature_name="item_id",
            negative_sampling_strategy=negative_sampling_strategy,
            num_negative_samples=2,
        )
    else:

        negative_sampler = None

    dataset = TwoTowerTrainingDataset(
        sequential_dataset,
        negative_sampler=negative_sampler,
        max_sequence_length=7,
    )
    return dataset


@pytest.fixture(scope="module")
def sequential_sample(twotower_training_dataset):
    return twotower_training_dataset.collate_fn([twotower_training_dataset[0], twotower_training_dataset[1]])


@pytest.fixture(scope="module")
def incorrect_sequential_sample(request, sequential_sample):
    sample = copy.deepcopy(sequential_sample)

    defect_type = request.param
    if defect_type == "missing field":
        sample.pop("padding_mask")
    elif defect_type == "wrong length":
        sample["feature_tensors"]["item_id"] = sample["feature_tensors"]["item_id"][:, 1:]
    elif defect_type == "index out of embedding":
        sample["feature_tensors"]["item_id"][0][-1] = 4
    else:
        raise ValueError(defect_type)
    return sample


@pytest.fixture(scope="module")
def twotower_train_dataloader(twotower_training_dataset):
    train_dataloader = torch.utils.data.DataLoader(
        twotower_training_dataset,
        batch_size=2,
        collate_fn=getattr(twotower_training_dataset, "collate_fn", None),
    )
    return train_dataloader


@pytest.fixture
def builder(tensor_schema, dummy_mapping_path, dummy_reference_path):
    query_tower_names = tensor_schema.names
    item_tower_names = ["item_id", "item_list_feature"]

    builder = (
        TwoTowerBuilder()
        .ecom(
            tensor_schema=tensor_schema,
            feature_mapping_path=dummy_mapping_path,
            item_reference_path=dummy_reference_path,
            hidden_size=64,
            head_count=1,
            block_count=1,
            seq_len=7,
            embedding_dropout_rate=0.2,
        )
        .query_tower_feature_names(query_tower_names)
        .item_tower_feature_names(item_tower_names)
    )
    return builder


@pytest.fixture
def model(builder):
    model = builder.build()
    return model


@pytest.fixture(
    params=[
        (CE, {"padding_idx": 3}),
        (CESampled, {"padding_idx": 3}),
        (BCE, {}),
        (BCESampled, {}),
        (LogInCE, {"vocab_size": 3}),
        (LogInCESampled, {}),
        (LogOutCE, {"padding_idx": 3, "vocab_size": 3}),
    ],
    ids=["CE loss", "CE sampled", "BCE", "BCE sampled", "LogInCE", "LogInCESampled", "LogOutCE"],
)
def model_parametrized(request, tensor_schema, dummy_mapping_path, dummy_reference_path):
    query_tower_names = tensor_schema.names
    item_tower_names = ["item_id", "item_list_feature"]

    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)

    model = (
        TwoTowerBuilder()
        .schema(tensor_schema)
        .embedder(
            SequentialEmbedder(
                tensor_schema,
                embed_size=64,
                categorical_list_feature_aggregation_method="sum",
            )
        )
        .attn_mask_builder(DefaultAttentionMaskBuilder(tensor_schema, 1))
        .query_tower_feature_names(query_tower_names)
        .item_tower_feature_names(item_tower_names)
        .query_embedding_aggregator(
            QueryTowerEmbeddingAggregator(
                SumAggregator(embedding_dim=64),
                embedding_dim=64,
                max_len=7,
                dropout=0.2,
            )
        )
        .item_embedding_aggregator(ConcatAggregator(embed_sizes=[64, 64], embedding_dim=64))
        .query_encoder(DiffTransformerBlock(hidden_size=64, num_heads=1, num_blocks=1))
        .query_tower_output_normalization(RMSNorm(embed_size=64))
        .item_encoder(SwiGLUEncoder(embed_size=64))
        .feature_mapping_path(dummy_mapping_path)
        .item_reference_path(dummy_reference_path)
        .loss(loss)
        .build()
    )
    return model
