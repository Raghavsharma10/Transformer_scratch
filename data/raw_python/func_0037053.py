def getPrimeFactors(n):
    """
    Get all the prime factor of given integer
    @param n integer
    @return list [1, ..., n]
    """
    lo = [1]
    n2 = n // 2
    k = 2
    for k in range(2, n2 + 1):
        if (n // k)*k == n:
            lo.append(k)
    return lo + [n, ]