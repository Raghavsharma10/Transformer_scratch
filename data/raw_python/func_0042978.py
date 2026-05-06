def binom(n, k):
    """
    Returns binomial coefficient (n choose k).
    """
    # http://blog.plover.com/math/choose.html
    if k > n:
        return 0
    if k == 0:
        return 1
    result = 1
    for denom in range(1, k + 1):
        result *= n
        result /= denom
        n -= 1
    return result