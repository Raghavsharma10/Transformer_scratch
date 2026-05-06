def string_to_number(s, alphabet):
    """
    Given a string ``s``, convert it to an integer composed of the given
    ``alphabet`` mapping, where the position of each element in ``alphabet`` is
    its radix value.

    Examples::

        >>> string_to_number('101111000110000101001110', '01')
        12345678

        >>> string_to_number('babbbbaaabbaaaababaabbba', 'ab')
        12345678

        >>> string_to_number('ZXP0', string.ascii_letters + string.digits)
        12345678

    """
    base = len(alphabet)
    inverse_alphabet = dict(zip(alphabet, xrange(0, base)))
    n = 0
    exp = 0
    for i in reversed(s):
        n += inverse_alphabet[i] * (base ** exp)
        exp += 1

    return n