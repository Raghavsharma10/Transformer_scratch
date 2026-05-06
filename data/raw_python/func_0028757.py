def scalarmult_B(e):
    """
    Implements scalarmult(B, e) more efficiently.
    """
    # scalarmult(B, l) is the identity
    e = e % l
    P = ident
    for i in range(253):
        if e & 1:
            P = edwards_add(P, Bpow[i])
        e = e // 2
    assert e == 0, e
    return P