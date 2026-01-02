from core.decoder_block import Decoderblock

def test_decoder_block_shape():
    T = 3
    d = 4

    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    block = Decoderblock(d)
    out = block.forward(X)

    assert len(out) == T
    assert len(out[0]) == d