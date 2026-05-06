def interpret_bits_value(val):
    """
    Converts input bits value from string to a single integer value or None.
    If a comma- or '+'-separated set of values are provided, they are summed.

    .. note::
        In order to flip the bits of the final result (after summation),
        for input of `str` type, prepend '~' to the input string. '~' must
        be prepended to the *entire string* and not to each bit flag!

    Parameters
    ----------
    val : int, str, None
        An integer bit mask or flag, `None`, or a comma- or '+'-separated
        string list of integer bit values. If `val` is a `str` and if
        it is prepended with '~', then the output bit mask will have its
        bits flipped (compared to simple sum of input val).

    Returns
    -------
    bitmask : int or None
        Returns and integer bit mask formed from the input bit value
        or `None` if input `val` parameter is `None` or an empty string.
        If input string value was prepended with '~', then returned
        value will have its bits flipped (inverse mask).

    """
    if isinstance(val, int) or val is None:
        return val

    else:
        val = str(val).strip()

        if val.startswith('~'):
            flip_bits = True
            val = val[1:].lstrip()
        else:
            flip_bits = False

        if val.startswith('('):
            if val.endswith(')'):
                val = val[1:-1].strip()
            else:
                raise ValueError('Unbalanced parantheses or incorrect syntax.')

        if ',' in val:
            valspl = val.split(',')
            bitmask = 0
            for v in valspl:
                bitmask += int(v)

        elif '+' in val:
            valspl = val.split('+')
            bitmask = 0
            for v in valspl:
                bitmask += int(v)

        elif val.upper() in ['', 'NONE', 'INDEF']:
            return None

        else:
            bitmask = int(val)

        if flip_bits:
            bitmask = ~bitmask

    return bitmask