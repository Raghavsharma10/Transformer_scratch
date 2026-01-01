from core.self_attention import selfAttention

def test_self_attention_shape():
    # T = 3 tokens, d = 4 dimensions
    T = 3
    d = 4

    X = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    sa = selfAttention(d)
    out = sa.forward(X)

    # shape checks
    assert len(out) == T
    assert len(out[0]) == d