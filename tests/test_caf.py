import torch

from fias.models.fusion import ContextAwareFusion


def test_caf_fuses_local_and_global_features():
    module = ContextAwareFusion(local_channels=64, global_channels=64, out_channels=64)
    local_feat = torch.randn(2, 64, 64, 64)
    global_feat = torch.randn(2, 64, 64, 64)

    fused = module(local_feat, global_feat)

    assert fused.shape == (2, 64, 64, 64)


def test_caf_output_changes_with_inputs():
    module = ContextAwareFusion(local_channels=32, global_channels=32, out_channels=32)
    local_feat = torch.ones(1, 32, 32, 32)
    global_feat = torch.zeros(1, 32, 32, 32)

    fused_a = module(local_feat, global_feat)
    fused_b = module(global_feat, local_feat)

    assert not torch.allclose(fused_a, fused_b)
