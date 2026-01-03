from core.encoder import Encoder

def test_encoder_shape():
    T = 3
    d = 4
    L = 2

    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    encoder = Encoder(L, d)
    out = encoder.forward(X)

    assert len(out) == T
    assert len(out[0]) == d