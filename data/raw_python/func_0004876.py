def length(string, until=None):
    """
    Returns the number of graphemes in the string.

    Note that this functions needs to traverse the full string to calculate the length,
    unlike `len(string)` and it's time consumption is linear to the length of the string
    (up to the `until` value).

    Only counts up to the `until` argument, if given. This is useful when testing
    the length of a string against some limit and the excess length is not interesting.

    >>> rainbow_flag = "🏳️‍🌈"
    >>> len(rainbow_flag)
    4
    >>> graphemes.length(rainbow_flag)
    1
    >>> graphemes.length("".join(str(i) for i in range(100)), 30)
    30
    """
    if until is None:
        return sum(1 for _ in GraphemeIterator(string))

    iterator = graphemes(string)
    count = 0
    while True:
        try:
            if count >= until:
                break
            next(iterator)
        except StopIteration:
            break
        else:
            count += 1

    return count