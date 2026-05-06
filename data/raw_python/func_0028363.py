def iterslice(iterable, start=0, stop=None, step=1):
    # type: (Iterable[T], int, int, int) -> Iterable[T]
    """ Like itertools.islice, but accept int and callables.

        If `start` is a callable, start the slice after the first time
        start(item) == True.

        If `stop` is a callable, stop the slice after the first time
        stop(item) == True.
    """

    if step < 0:
        raise ValueError("The step can not be negative: '%s' given" % step)

    if not isinstance(start, int):

        # [Callable:Callable]
        if not isinstance(stop, int) and stop:
            return stops_when(starts_when(iterable, start), stop)

        # [Callable:int]
        return starts_when(itertools.islice(iterable, None, stop, step), start)

    # [int:Callable]
    if not isinstance(stop, int) and stop:
        return stops_when(itertools.islice(iterable, start, None, step), stop)

    # [int:int]
    return itertools.islice(iterable, start, stop, step)