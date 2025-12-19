import pytest

import torch
import lightning as L
from replay.nn.lightning import LightningModule


@pytest.mark.torch
def test_training_sasrec_with_different_losses(model_parametrized, parquet_module):
    sasrec = LightningModule(model_parametrized)
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(sasrec, datamodule=parquet_module)


@pytest.mark.torch
def test_sasrec_checkpoining(model, parquet_module, tmp_path):
    sasrec = LightningModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    ckpt_path = tmp_path / "checkpoints/last.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_sasrec = LightningModule.load_from_checkpoint(ckpt_path, model=model)

    sasrec.eval()
    loaded_sasrec.eval()

    batch = parquet_module.compiled_transforms["train"](next(iter(parquet_module.train_dataloader())))
    output1 = sasrec(batch)
    output2 = loaded_sasrec(batch)

    torch.testing.assert_close(output1.logits, output2.logits)
    torch.testing.assert_close(output1.hidden_states[0], output2.hidden_states[0])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.LongTensor([1]), torch.LongTensor([1, 2]), None],
)
def test_sasrec_prediction_with_candidates(model, parquet_module, candidates_to_score):
    sasrec = LightningModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    sasrec.candidates_to_score = candidates_to_score
    trainer = L.Trainer(inference_mode=True)
    predictions = trainer.predict(sasrec, datamodule=parquet_module)

    if candidates_to_score is not None:
        assert torch.equal(sasrec.candidates_to_score, candidates_to_score)
    else:
        assert sasrec.candidates_to_score is None

    for pred in predictions:
        if candidates_to_score is None:
            assert pred.size() == (1, 3)
        elif isinstance(candidates_to_score, torch.BoolTensor):
            assert pred.size() == (1, candidates_to_score.sum())
        else:
            assert pred.size() == (1, candidates_to_score.shape[0])


@pytest.mark.torch
def test_predictions_sasrec_equal_with_permuted_candidates(model, parquet_module):
    sasrec = LightningModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    sorted_candidates = torch.LongTensor([0, 0, 1, 1, 1, 2])
    permuted_candidates = torch.LongTensor([1, 0, 2, 1, 1, 0])
    _, ordering = torch.sort(permuted_candidates)

    trainer = L.Trainer(inference_mode=True)

    sasrec.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(sasrec, datamodule=parquet_module)

    sasrec.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(sasrec, datamodule=parquet_module)

    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(predictions_permuted_candidates[i][:, ordering], predictions_sorted_candidates[i])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.FloatTensor([1]), torch.BoolTensor([1, 0])],
)
def test_sasrec_prediction_invalid_candidates_to_score(
    model, parquet_module, candidates_to_score
):
    sasrec = LightningModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    trainer = L.Trainer(inference_mode=True)
    sasrec.candidates_to_score = candidates_to_score

    with pytest.raises(RuntimeError):
        trainer.predict(sasrec, datamodule=parquet_module)
