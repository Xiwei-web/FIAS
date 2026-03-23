import torch

from fias.models.decoders import MixAttentionDecoder


def test_mixatt_decoder_output_shapes():
    decoder = MixAttentionDecoder(channels=(32, 64, 128, 256))
    fused_features = [
        torch.randn(2, 32, 256, 256),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
    ]

    output, decoder_features = decoder(fused_features)

    assert output.shape == (2, 32, 256, 256)
    assert len(decoder_features) == 4
    assert decoder_features[0].shape == (2, 256, 32, 32)
    assert decoder_features[1].shape == (2, 128, 64, 64)
    assert decoder_features[2].shape == (2, 64, 128, 128)
    assert decoder_features[3].shape == (2, 32, 256, 256)


def test_mixatt_decoder_backpropagates():
    decoder = MixAttentionDecoder(channels=(16, 32, 64, 128))
    fused_features = [
        torch.randn(1, 16, 128, 128, requires_grad=True),
        torch.randn(1, 32, 64, 64, requires_grad=True),
        torch.randn(1, 64, 32, 32, requires_grad=True),
        torch.randn(1, 128, 16, 16, requires_grad=True),
    ]

    output, _ = decoder(fused_features)
    output.mean().backward()

    assert fused_features[0].grad is not None
