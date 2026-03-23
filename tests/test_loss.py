import torch

from fias.losses import CrossEntropySegLoss, DiceLoss, FeatureMixingSegLoss


def test_dice_loss_returns_scalar():
    criterion = DiceLoss()
    logits = torch.randn(2, 4, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))

    loss = criterion(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cross_entropy_loss_returns_scalar():
    criterion = CrossEntropySegLoss()
    logits = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 3, (2, 32, 32))

    loss = criterion(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_feature_mixing_loss_uses_main_and_aux_predictions():
    criterion = FeatureMixingSegLoss(gamma=0.4)
    outputs = {
        "logits": torch.randn(2, 4, 64, 64),
        "aux_logits": [
            torch.randn(2, 4, 64, 64),
            torch.randn(2, 4, 64, 64),
            torch.randn(2, 4, 64, 64),
        ],
    }
    target = torch.randint(0, 4, (2, 64, 64))

    loss = criterion(outputs, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
