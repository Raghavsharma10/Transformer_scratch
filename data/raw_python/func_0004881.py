def safe_split_index(string, max_len):
    """
    Returns the highest index up to `max_len` at which the given string can be sliced, without breaking a grapheme.

    This is useful for when you want to split or take a substring from a string, and don't really care about
    the exact grapheme length, but don't want to risk breaking existing graphemes.

    This function does normally not traverse the full grapheme sequence up to the given length, so it can be used
    for arbitrarily long strings and high `max_len`s. However, some grapheme boundaries depend on the previous state,
    so the worst case performance is O(n). In practice, it's only very long non-broken sequences of country flags
    (represented as Regional Indicators) that will perform badly.

    The return value will always be between `0` and `len(string)`.

    >>> string = "tamil நி (ni)"
    >>> i = grapheme.safe_split_index(string, 7)
    >>> i
    6
    >>> string[:i]
    'tamil '
    >>> string[i:]
    'நி (ni)'
    """
    last_index = get_last_certain_break_index(string, max_len)
    for l in grapheme_lengths(string[last_index:]):
        if last_index + l > max_len:
            break
        last_index += l
    return last_index