def tuples(stream, *keys):
    """Reformat data as tuples.

    Parameters
    ----------
    stream : iterable
        Stream of data objects.

    *keys : strings
        Keys to use for ordering data.

    Yields
    ------
    items : tuple of np.ndarrays
        Data object reformated as a tuple.

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    KeyError
        If a data object does not contain the requested key.
    """
    if not keys:
        raise PescadorError('Unable to generate tuples from '
                            'an empty item set')
    for data in stream:
        try:
            yield tuple(data[key] for key in keys)
        except TypeError:
            raise DataError("Malformed data stream: {}".format(data))