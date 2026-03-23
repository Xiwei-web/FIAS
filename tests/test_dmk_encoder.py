import torch

from fias.models.encoders import DMKEncoder


def test_dmk_encoder_returns_four_feature_maps():
    model = DMKEncoder(in_channels=1, channels=(32, 64, 128, 256))
    x = torch.randn(2, 1, 256, 256)

    features = model(x)

    assert len(features) == 4
    assert features[0].shape == (2, 32, 256, 256)
    assert features[1].shape == (2, 64, 128, 128)
    assert features[2].shape == (2, 128, 64, 64)
    assert features[3].shape == (2, 256, 32, 32)


def test_dmk_encoder_backpropagates():
    model = DMKEncoder(in_channels=1, channels=(16, 32, 64, 128))
    x = torch.randn(1, 1, 128, 128, requires_grad=True)

    features = model(x)
    loss = sum(feature.mean() for feature in features)
    loss.backward()

    assert x.grad is not None
