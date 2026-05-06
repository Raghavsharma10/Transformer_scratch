def ipshuffle(l, random=None):
    r"""Shuffle list `l` inplace and return it."""
    import random as _random
    _random.shuffle(l, random)
    return l