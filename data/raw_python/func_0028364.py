def firsts(iterable, items=1, default=None):
    # type: (Iterable[T], int, T) -> Iterable[T]
    """ Lazily return the first x items from this iterable or default. """

    try:
        items = int(items)
    except (ValueError, TypeError):
        raise ValueError("items should be usable as an int but is currently "
                         "'{}' of type '{}'".format(items, type(items)))

    # TODO: replace this so that it returns lasts()
    if items < 0:
        raise ValueError(ww.f("items is {items} but should "
                              "be greater than 0. If you wish to get the last "
                              "items, use the lasts() function."))

    i = 0
    for i, item in zip(range(items), iterable):
        yield item

    for x in range(items - (i + 1)):
        yield default