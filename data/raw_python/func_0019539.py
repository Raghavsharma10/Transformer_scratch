def buffer_stream(stream, buffer_size, partial=False, axis=None):
    '''Buffer "data" from an stream into one data object.

    Parameters
    ----------
    stream : stream
        The stream to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    partial : bool, default=False
        If True, yield a final partial batch on under-run.

    axis : int or None
        If `None` (default), concatenate data along a new 0th axis.
        Otherwise, concatenation is performed along the specified axis.

        This is primarily useful when combining data that already has a
        dimension for buffer index, e.g., when buffering buffers.

    Yields
    ------
    batch
        A batch of size at most `buffer_size`

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    '''

    data = []
    count = 0

    for item in stream:
        data.append(item)
        count += 1

        if count < buffer_size:
            continue
        try:
            yield __stack_data(data, axis=axis)
        except (TypeError, AttributeError):
            raise DataError("Malformed data stream: {}".format(data))
        finally:
            data = []
            count = 0

    if data and partial:
        yield __stack_data(data, axis=axis)