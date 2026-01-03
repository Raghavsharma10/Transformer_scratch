from core.decoder import Decoder

def test_decoder_shape():
    T = 3
    d = 4
    L = 2

    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    decoder = Decoder(L, d)
    out = decoder.forward(X)

    assert len(out) == T
    assert len(out[0]) == d