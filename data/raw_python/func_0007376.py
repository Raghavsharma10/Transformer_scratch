def number_to_string(n, alphabet):
    """
    Given an non-negative integer ``n``, convert it to a string composed of
    the given ``alphabet`` mapping, where the position of each element in
    ``alphabet`` is its radix value.

    Examples::

        >>> number_to_string(12345678, '01')
        '101111000110000101001110'

        >>> number_to_string(12345678, 'ab')
        'babbbbaaabbaaaababaabbba'

        >>> number_to_string(12345678, string.ascii_letters + string.digits)
        'ZXP0'

        >>> number_to_string(12345, ['zero ', 'one ', 'two ', 'three ', 'four ', 'five ', 'six ', 'seven ', 'eight ', 'nine '])
        'one two three four five '

    """
    result = ''
    base = len(alphabet)
    current = int(n)
    if current < 0:
        raise ValueError("invalid n (must be non-negative): %s", n)
    while current:
        result = alphabet[current % base] + result
        current = current // base

    return result