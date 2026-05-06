def publickey_unsafe(sk):
    """
    Not safe to use with secret keys or secret data.

    See module docstring.  This function should be used for testing only.
    """
    h = H(sk)
    a = 2 ** (b - 2) + sum(2 ** i * bit(h, i) for i in range(3, b - 2))
    A = scalarmult_B(a)
    return encodepoint(A)