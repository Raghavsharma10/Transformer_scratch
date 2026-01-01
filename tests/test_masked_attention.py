from core.attention import scaled_dot_product_attention, causal_mask

def test_causal_mask_blocks_future():
    Q = [[1,0],[0,1]]
    K = [[1,0],[0,1]]
    V = [[1,0],[0,1]]

    mask = causal_mask(2)
    out = scaled_dot_product_attention(Q, K, V, mask=mask)

    # token 0 must not attend to token 1
    assert abs(out[0][1]) < 1e-6