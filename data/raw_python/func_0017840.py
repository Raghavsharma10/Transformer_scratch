def coerce_data_source(x):
    """
    Helper function to coerce an object into a data source, selecting the
    appropriate data source class for the given object. If `x` is already
    a data source it is returned as is.

    Parameters
    ----------
    x: any
        The object to coerce. If `x` is a data source, it is returned as is.
        If it is a list or tuple of array-like objects they will be wrapped
        in an `ArrayDataSource` that will be returned. If `x` is an iterator
        it will be wrapped in an `IteratorDataSource`. If it is a callable
        it will be wrapped in a `CallableDataSource`.

    Returns
    -------
    `x` coerced into a data source

    Raises
    ------
    `TypeError` if `x` is not a data souce, a list or tuple of array-like
    objects, an iterator or a callable.
    """
    if isinstance(x, AbstractDataSource):
        return x
    elif isinstance(x, (list, tuple)):
        # Sequence of array-likes
        items = []
        for item in x:
            if _is_array_like(item):
                items.append(item)
            else:
                raise TypeError(
                    'Cannot convert x to a data source; x is a sequence and '
                    'one of the elements is not an array-like object, rather '
                    'a {}'.format(type(item)))
        if len(items) == 0:
            raise ValueError('Cannot convert x to a data source; x is an '
                             'empty sequence')
        return ArrayDataSource(items)
    elif isinstance(x, collections.Iterator):
        return IteratorDataSource(x)
    elif callable(x):
        return CallableDataSource(x)
    else:
        raise TypeError('Cannot convert x to a data source; can only handle '
                        'iterators, callables, non-empty sequences of '
                        'array-like objects; cannot '
                        'handle {}'.format(type(x)))