import torch

from fias.models import FIASModel


def test_fias_model_forward_shapes():
    model = FIASModel(in_channels=1, num_classes=9, channels=(32, 64, 128, 256))
    x = torch.randn(2, 1, 256, 256)

    outputs = model(x)

    assert "logits" in outputs
    assert "aux_logits" in outputs
    assert "fused_features" in outputs
    assert outputs["logits"].shape == (2, 9, 256, 256)
    assert len(outputs["aux_logits"]) == 3
    assert len(outputs["fused_features"]) == 4


def test_fias_model_backward_pass():
    model = FIASModel(in_channels=1, num_classes=4, channels=(16, 32, 64, 128))
    x = torch.randn(1, 1, 128, 128, requires_grad=True)

    outputs = model(x)
    loss = outputs["logits"].mean()
    loss.backward()

    assert x.grad is not None
