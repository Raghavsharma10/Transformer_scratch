from core.attention import scaled_dot_product_attention

def test_attention_shape():

    Q = [[1,0],[0,1]]
    K = [[1,0],[0,1]]
    V = [[5,5],[10,10]]

    out = scaled_dot_product_attention(Q,K,V)

    assert len(out) == 2
    assert(len(out[0])) == 2