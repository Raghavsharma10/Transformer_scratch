def round_(values, decimals=None, width=0,
           lfill=None, rfill=None, **kwargs):
    """Prints values with a maximum number of digits in doctests.

    See the documentation on function |repr| for more details.  And
    note thate the option keyword arguments are passed to the print function.

    Usually one would apply function |round_| on a single or a vector
    of numbers:

    >>> from hydpy import round_
    >>> round_(1./3., decimals=6)
    0.333333
    >>> round_((1./2., 1./3., 1./4.), decimals=4)
    0.5, 0.3333, 0.25

    Additionally, one can supply a `width` and a `rfill` argument:
    >>> round_(1.0, width=6, rfill='0')
    1.0000

    Alternatively, one can use the `lfill` arguments, which
    might e.g. be usefull for aligning different strings:

    >>> round_('test', width=6, lfill='_')
    __test

    Using both the `lfill` and the `rfill` argument raises an error:

    >>> round_(1.0, lfill='_', rfill='0')
    Traceback (most recent call last):
    ...
    ValueError: For function `round_` values are passed for both \
arguments `lfill` and `rfill`.  This is not allowed.
    """
    if decimals is None:
        decimals = hydpy.pub.options.reprdigits
    with hydpy.pub.options.reprdigits(decimals):
        if isinstance(values, abctools.IterableNonStringABC):
            string = repr_values(values)
        else:
            string = repr_(values)
        if (lfill is not None) and (rfill is not None):
            raise ValueError(
                'For function `round_` values are passed for both arguments '
                '`lfill` and `rfill`.  This is not allowed.')
        if (lfill is not None) or (rfill is not None):
            width = max(width, len(string))
            if lfill is not None:
                string = string.rjust(width, lfill)
            else:
                string = string.ljust(width, rfill)
        print(string, **kwargs)