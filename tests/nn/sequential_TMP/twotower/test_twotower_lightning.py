from contextlib import nullcontext as no_exception

import pytest

from amazme.replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from amazme.replay.models.nn import LightingModule
    from amazme.replay.models.nn.loss import BCESampled, CESampled, LogInCESampled
    from amazme.replay.models.nn.optimizer_utils import (
        FatLRSchedulerFactory,
        FatOptimizerFactory,
        LambdaLRSchedulerFactory,
    )

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "twotower_training_dataset",
    [
        (False, None),
        (True, "global_uniform"),
    ],
    ids=["without negative sampler", "with negative sampler"],
    indirect=True,
)
def test_training_twotower_with_different_losses(model_parametrized, twotower_train_dataloader):
    twotower = LightingModule(model_parametrized, lr_scheduler_factory=LambdaLRSchedulerFactory(warmup_steps=1))
    trainer = L.Trainer(max_epochs=2)
    if twotower_train_dataloader.dataset.negative_sampler is None and isinstance(
        model_parametrized.loss, (BCESampled, CESampled, LogInCESampled)
    ):
        with pytest.raises(AssertionError):
            trainer.fit(twotower, twotower_train_dataloader)
    else:
        trainer.fit(twotower, twotower_train_dataloader)


@pytest.mark.torch
def test_twotower_checkpoining(model, twotower_train_dataloader, tmp_path):
    twotower = LightingModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, twotower_train_dataloader)

    ckpt_path = tmp_path / "checkpoints/last.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_twotower = LightingModule.load_from_checkpoint(ckpt_path, model=model)

    twotower.eval()
    loaded_twotower.eval()

    batch = next(iter(twotower_train_dataloader))
    output1 = twotower(batch)
    output2 = loaded_twotower(batch)

    torch.testing.assert_close(output1.logits, output2.logits)
    torch.testing.assert_close(output1.hidden_states[0], output2.hidden_states[0])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.LongTensor([1]), torch.LongTensor([1, 2]), None, torch.BoolTensor([0, 1, 0])],
)
def test_twotower_prediction_with_candidates(model, twotower_train_dataloader, test_dataloader, candidates_to_score):
    twotower = LightingModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, twotower_train_dataloader)

    twotower.candidates_to_score = candidates_to_score
    trainer = L.Trainer(inference_mode=True)
    predictions = trainer.predict(twotower, test_dataloader)

    if candidates_to_score is not None:
        assert torch.equal(twotower.candidates_to_score, candidates_to_score)
    else:
        assert twotower.candidates_to_score is None

    for pred in predictions:
        if candidates_to_score is None:
            assert pred.size() == (1, 3)
        elif isinstance(candidates_to_score, torch.BoolTensor):
            assert pred.size() == (1, candidates_to_score.sum())
        else:
            assert pred.size() == (1, candidates_to_score.shape[0])


@pytest.mark.torch
def test_predictions_twotower_equal_with_permuted_candidates(model, twotower_train_dataloader, test_dataloader):
    twotower = LightingModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, twotower_train_dataloader)

    sorted_candidates = torch.LongTensor([0, 0, 1, 1, 1, 2])
    permuted_candidates = torch.LongTensor([1, 0, 2, 1, 1, 0])
    _, ordering = torch.sort(permuted_candidates)

    trainer = L.Trainer(inference_mode=True)

    twotower.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(twotower, test_dataloader)

    twotower.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(twotower, test_dataloader)

    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(predictions_permuted_candidates[i][:, ordering], predictions_sorted_candidates[i])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.FloatTensor([1]), torch.BoolTensor([1, 0])],
)
def test_twotower_prediction_invalid_candidates_to_score(
    model, twotower_train_dataloader, test_dataloader, candidates_to_score
):
    twotower = LightingModule(model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, twotower_train_dataloader)

    trainer = L.Trainer(inference_mode=True)
    twotower.candidates_to_score = candidates_to_score

    with pytest.raises(IndexError):
        trainer.predict(twotower, test_dataloader)


@pytest.mark.torch
@pytest.mark.parametrize(
    "optimizer_factory, lr_scheduler_factory",
    [
        (None, None),
        (FatOptimizerFactory(), None),
        (None, FatLRSchedulerFactory()),
        (FatOptimizerFactory(), FatLRSchedulerFactory()),
        (None, LambdaLRSchedulerFactory(warmup_steps=6)),
        (FatOptimizerFactory(), LambdaLRSchedulerFactory(warmup_steps=6)),
    ],
)
def test_twotower_configure_optimizers(model, optimizer_factory, lr_scheduler_factory):
    twotower = LightingModule(
        model,
        lr_scheduler_factory=lr_scheduler_factory,
        optimizer_factory=optimizer_factory,
    )

    parameters = twotower.configure_optimizers()
    if isinstance(parameters, tuple):
        assert isinstance(parameters[0][0], torch.optim.Adam)
        if isinstance(lr_scheduler_factory, FatLRSchedulerFactory):
            assert isinstance(parameters[1][0], torch.optim.lr_scheduler.StepLR)
        if isinstance(lr_scheduler_factory, LambdaLRSchedulerFactory):
            assert isinstance(parameters[1][0]["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    else:
        assert isinstance(parameters, torch.optim.Adam)


@pytest.mark.torch
def test_twotower_get_set_optim_factory(model):
    optim_factory = FatOptimizerFactory()
    twotower = LightingModule(model, optimizer_factory=optim_factory)

    assert twotower._optimizer_factory is optim_factory
    new_factory = FatOptimizerFactory(learning_rate=0.1)
    twotower._optimizer_factory = new_factory
    assert twotower._optimizer_factory is new_factory


@pytest.mark.torch
@pytest.mark.parametrize(
    "warmup_lr, normal_lr, expected_exception",
    [
        (-1.0, 0.1, pytest.raises(ValueError)),
        (1.0, -0.1, pytest.raises(ValueError)),
        (0.0, 0.0, pytest.raises(ValueError)),
        (0.1, 0.01, no_exception()),
        (0.1, 1.0, no_exception()),
    ],
)
def test_configure_lambda_lr_scheduler(warmup_lr, normal_lr, expected_exception):
    with expected_exception:
        LambdaLRSchedulerFactory(warmup_steps=1, warmup_lr=warmup_lr, normal_lr=normal_lr)
