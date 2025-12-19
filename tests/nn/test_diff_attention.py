import pytest
import torch
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential.sasrec.diff_transformer import DiffTransformerLayer


@pytest.mark.torch
def test_diff_attention_forward(tensor_schema, simple_batch):
    mask_block = DefaultAttentionMask(reference_feature_name="item_id", num_heads=2)
    diff_attn_block = DiffTransformerLayer(embedding_dim=64, num_heads=2, num_blocks=2)

    embedding_shape = (*simple_batch["feature_tensors"]["item_id"].shape,
                        tensor_schema["item_id"].embedding_dim
    )
    mask = mask_block(simple_batch["feature_tensors"], simple_batch["padding_mask"])

    attn_hidden = diff_attn_block(
        feature_tensors=simple_batch["feature_tensors"],
        input_embeddings=torch.rand(embedding_shape),
        padding_mask=simple_batch["padding_mask"],
        attention_mask=mask,
    )
    assert attn_hidden.shape == embedding_shape

    # mask_reshaped = mask.view(4, 2, 5, 5)

    # attn_hidden_with_mask_reshaped = diff_attn_block(
    #     feature_tensors=simple_batch["feature_tensors"],
    #     input_embeddings=torch.rand(embedding_shape),
    #     padding_mask=None,
    #     attention_mask=mask_reshaped,
    # )
    # assert attn_hidden_with_mask_reshaped.shape == embedding_shape
