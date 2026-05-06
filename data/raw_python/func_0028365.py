def lasts(iterable, items=1, default=None):
    # type: (Iterable[T], int, T) -> Iterable[T]
    """ Lazily return the last x items from this iterable or default. """

    last_items = deque(iterable, maxlen=items)

    for _ in range(items - len(last_items)):
        yield default

    for y in last_items:
        yield y