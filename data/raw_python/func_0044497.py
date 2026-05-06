def _sum(data):
    """_sum(data [, start]) -> value
    Return a high-precision sum of the given numeric data. If optional
    argument ``start`` is given, it is added to the total. If ``data`` is
    empty, ``start`` (defaulting to 0) is returned.
    Examples
    --------
    >>> _sum([3, 2.25, 4.5, -0.5, 1.0], 0.75)
    11.0
    Some sources of round-off error will be avoided:
    >>> _sum([1e50, 1, -1e50] * 1000)  # Built-in sum returns zero.
    1000.0
    Fractions and Decimals are also supported:
    >>> from fractions import Fraction as F
    >>> _sum([F(2, 3), F(7, 5), F(1, 4), F(5, 6)])
    Fraction(63, 20)
    >>> from decimal import Decimal as D
    >>> data = [D("0.1375"), D("0.2108"), D("0.3061"), D("0.0419")]
    >>> _sum(data)
    Decimal('0.6963')
    Mixed types are currently treated as an error, except that int is
    allowed.
    """
    data = iter(data)
    n = _first(data)

    if n is not None:
        data = chain([n], data)
        if isinstance(n, F):
            return math.fsum(data)
        return sum(data)
    return 0