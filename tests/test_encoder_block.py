from core.encoder_block import Encoderblock

def test_encoder_block_shape():
    T = 3
    d = 4

    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    block = Encoderblock(d)
    out = block.forward(X)

    assert len(out) == T
    assert len(out[0]) == d