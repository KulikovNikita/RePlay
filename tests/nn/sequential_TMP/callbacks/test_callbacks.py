import pytest

from amazme.replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from amazme.replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionDataset
    from amazme.replay.models.nn.sequential.callbacks import ValidationMetricsCallback
    from amazme.replay.models.nn.sequential.postprocessors import RemoveSeenItems

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
def test_empty_metrics_passed():
    callback = ValidationMetricsCallback(metrics=["coverage", "unseen-precision"], ks=[1], item_count=1)

    assert callback._separate_metrics(None) == (None, None)


@pytest.mark.torch
@pytest.mark.parametrize(
    "metrics, postprocessor",
    [
        (["coverage", "unseen-precision"], RemoveSeenItems),
        (["coverage"], RemoveSeenItems),
        (["coverage", "unseen-precision"], None),
        (["coverage"], None),
    ],
)
def test_validation_callbacks(item_user_sequential_dataset, train_loader, val_loader, metrics, postprocessor):
    callback = ValidationMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=1,
        postprocessors=[postprocessor(item_user_sequential_dataset)] if postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    if any([metric_name.startswith("unseen") for metric_name in metrics]):
        with pytest.raises(AttributeError):
            trainer.fit(model, train_loader, val_loader)
        return
    else:
        trainer.fit(model, train_loader, val_loader)

    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)
