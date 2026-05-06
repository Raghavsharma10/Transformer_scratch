def first_true(iterable, func):
    # type: (Iterable[T], Callable) -> T
    """" Return the first item of the iterable for which func(item) == True.

        Or raises IndexError.

        WARNING: this will consume generators.
    """
    try:
        return next((x for x in iterable if func(x)))
    except StopIteration as e:
        # TODO: Find a better error message
        raise_from(IndexError('No match for %s' % func), e)