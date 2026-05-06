def head_tail_middle(src):
    """Returns a tuple consisting of the head of a enumerable, the middle
    as a list and the tail of the enumerable. If the enumerable is 1 item, the
    middle will be empty and the tail will be None. 

    >>> head_tail_middle([1, 2, 3, 4])
    1, [2, 3], 4
    """

    if len(src) == 0:
        return None, [], None

    if len(src) == 1:
        return src[0], [], None

    if len(src) == 2:
        return src[0], [], src[1]

    return src[0], src[1:-1], src[-1]