def _nbOperations(n):
    """
    Exact number of atomic operations in _radixPass.

    """
    if n < 2:
        return 0
    else:
        n0 = (n + 2) // 3
        n02 = n0 + n // 3
        return 3 * (n02) + n0 + _nbOperations(n02)