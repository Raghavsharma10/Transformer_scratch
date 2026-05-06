def _slice(start, stop, step):
    """
    Generate pairs so that you can slice from start to stop, step elements at a time
    :param start: The start of the generated series
    :param stop: The last of the generated series
    :param step: The difference between the first element of the returned pair and the second
    :return: A pair that you can use to slice
    """
    if step == 0:
        raise ValueError("slice() arg 3 must not be zero")
    if start==stop:
        raise StopIteration

    previous = start
    next = start + step
    while next < stop:
        yield previous, next
        previous += step
        next += step
    yield previous, stop