def dollars_to_cents(s, allow_negative=False):
    """
    Given a string or integer representing dollars, return an integer of
    equivalent cents, in an input-resilient way.
    
    This works by stripping any non-numeric characters before attempting to
    cast the value.

    Examples::

        >>> dollars_to_cents('$1')
        100
        >>> dollars_to_cents('1')
        100
        >>> dollars_to_cents(1)
        100
        >>> dollars_to_cents('1e2')
        10000
        >>> dollars_to_cents('-1$', allow_negative=True)
        -100
        >>> dollars_to_cents('1 dollar')
        100
    """
    # TODO: Implement cents_to_dollars
    if not s:
        return

    if isinstance(s, string_types):
        s = ''.join(RE_NUMBER.findall(s))

    dollars = int(round(float(s) * 100))
    if not allow_negative and dollars < 0:
        raise ValueError('Negative values not permitted.')

    return dollars