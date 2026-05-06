def sum(data, start=0):
    """sum(data [, start]) -> value

    Return a high-precision sum of the given numeric data. If optional
    argument ``start`` is given, it is added to the total. If ``data`` is
    empty, ``start`` (defaulting to 0) is returned.
    """
    n, d = exact_ratio(start)
    T = type(start)
    partials = {d: n}  # map {denominator: sum of numerators}
    # Micro-optimizations.
    coerce_types_ = coerce_types
    exact_ratio_ = exact_ratio
    partials_get = partials.get
    # Add numerators for each denominator, and track the "current" type.
    for x in data:
        T = coerce_types_(T, type(x))
        n, d = exact_ratio_(x)
        partials[d] = partials_get(d, 0) + n
    if None in partials:
        assert issubclass(T, (float, Decimal))
        assert not isfinite(partials[None])
        return T(partials[None])
    total = Fraction()
    for d, n in sorted(partials.items()):
        total += Fraction(n, d)
    if issubclass(T, int):
        assert total.denominator == 1
        return T(total.numerator)
    if issubclass(T, Decimal):
        return T(total.numerator) / total.denominator
    return T(total)