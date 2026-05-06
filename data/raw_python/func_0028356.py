def starts_when(iterable, condition):
    # type: (Iterable, Union[Callable, Any]) -> Iterable
    """Start yielding items when a condition arise.

    Args:
        iterable: the iterable to filter.
        condition: if the callable returns True once, start yielding
                   items. If it's not a callable, it will be converted
                   to one as `lambda condition: condition == item`.

    Example:

        >>> list(starts_when(range(10), lambda x: x > 5))
        [6, 7, 8, 9]
        >>> list(starts_when(range(10), 7))
        [7, 8, 9]
    """
    if not callable(condition):
        cond_value = condition

        def condition(x):
            return x == cond_value
    return itertools.dropwhile(lambda x: not condition(x), iterable)