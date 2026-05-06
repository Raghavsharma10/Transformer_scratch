def get_hexagram(method='THREE COIN'):
    """
    Return one or two hexagrams using any of a variety of divination methods.

    The ``NAIVE`` method simply returns a uniformally random ``int`` between
    ``1`` and ``64``.

    All other methods return a 2-tuple where the first value
    represents the starting hexagram and the second represents the 'moving to'
    hexagram.

    To find the name and unicode glyph for a found hexagram, look it up in
    the module-level `hexagrams` dict.

    Args:
        method (str): ``'THREE COIN'``, ``'YARROW'``, or ``'NAIVE'``,
            the divination method model to use. Note that the three coin and
            yarrow methods are not actually literally simulated,
            but rather statistical models reflecting the methods are passed
            to `blur.rand` functions to accurately approximate them.

    Returns:
        int: If ``method == 'NAIVE'``, the ``int`` key of the found hexagram.
        Otherwise a `tuple` will be returned.

        tuple: A 2-tuple of form ``(int, int)``  where the first value
        is key of the starting hexagram and the second is that of the
        'moving-to' hexagram.

    Raises: ValueError if ``method`` is invalid

    Examples:

    The function being used alone: ::

        >>> get_hexagram(method='THREE COIN')                  # doctest: +SKIP
        # Might be...
        (55, 2)
        >>> get_hexagram(method='YARROW')                      # doctest: +SKIP
        # Might be...
        (41, 27)
        >>> get_hexagram(method='NAIVE')                       # doctest: +SKIP
        # Might be...
        26

    Usage in combination with hexagram lookup: ::

        >>> grams = get_hexagram()
        >>> grams                                              # doctest: +SKIP
        (47, 42)
        # unpack hexagrams for convenient reference
        >>> initial, moving_to = grams
        >>> hexagrams[initial]                                 # doctest: +SKIP
        ('䷮', '困', 'Confining')
        >>> hexagrams[moving_to]                               # doctest: +SKIP
        ('䷩', '益', 'Augmenting')
        >>> print('{} moving to {}'.format(
        ...     hexagrams[initial][2],
        ...     hexagrams[moving_to][2])
        ...     )                                              # doctest: +SKIP
        Confining moving to Augmenting
    """
    if method == 'THREE COIN':
        weights = [('MOVING YANG', 2),
                   ('MOVING YIN',  2),
                   ('STATIC YANG', 6),
                   ('STATIC YIN',  6)]
    elif method == 'YARROW':
        weights = [('MOVING YANG', 8),
                   ('MOVING YIN',  2),
                   ('STATIC YANG', 11),
                   ('STATIC YIN',  17)]
    elif method == 'NAIVE':
        return random.randint(1, 64)
    else:
        raise ValueError('`method` value of "{}" is invalid')

    hexagram_1 = []
    hexagram_2 = []

    for i in range(6):
        roll = weighted_choice(weights)
        if roll == 'MOVING YANG':
            hexagram_1.append(1)
            hexagram_2.append(0)
        elif roll == 'MOVING YIN':
            hexagram_1.append(0)
            hexagram_2.append(1)
        elif roll == 'STATIC YANG':
            hexagram_1.append(1)
            hexagram_2.append(1)
        else:  # if roll == 'STATIC YIN'
            hexagram_1.append(0)
            hexagram_2.append(0)
    # Convert hexagrams lists into tuples
    hexagram_1 = tuple(hexagram_1)
    hexagram_2 = tuple(hexagram_2)
    return (_hexagram_dict[hexagram_1], _hexagram_dict[hexagram_2])