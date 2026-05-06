def assignrepr_values2(values, prefix):
    """Return a prefixed and properly aligned string representation
    of the given 2-dimensional value matrix using function |repr|.

    >>> from hydpy.core.objecttools import assignrepr_values2
    >>> import numpy
    >>> print(assignrepr_values2(numpy.eye(3), 'test(') + ')')
    test(1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0)

    Functions |assignrepr_values2| works also on empty iterables:

    >>> print(assignrepr_values2([[]], 'test(') + ')')
    test()
    """
    lines = []
    blanks = ' '*len(prefix)
    for (idx, subvalues) in enumerate(values):
        if idx == 0:
            lines.append('%s%s,' % (prefix, repr_values(subvalues)))
        else:
            lines.append('%s%s,' % (blanks, repr_values(subvalues)))
    lines[-1] = lines[-1][:-1]
    return '\n'.join(lines)