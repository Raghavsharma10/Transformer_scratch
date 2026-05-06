def stops_when(iterable, condition):
    # type: (Iterable, Union[Callable, Any]) -> Iterable
    """Stop yielding items when a condition arise.

    Args:
        iterable: the iterable to filter.
        condition: if the callable returns True once, stop yielding
                   items. If it's not a callable, it will be converted
                   to one as `lambda condition: condition == item`.

    Example:

        >>> list(stops_when(range(10), lambda x: x > 5))
        [0, 1, 2, 3, 4, 5]
        >>> list(stops_when(range(10), 7))
        [0, 1, 2, 3, 4, 5, 6]
    """
    if not callable(condition):
        cond_value = condition

        def condition(x):
            return x == cond_value
    return itertools.takewhile(lambda x: not condition(x), iterable)