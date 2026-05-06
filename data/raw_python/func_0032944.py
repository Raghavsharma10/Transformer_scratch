def first_completed(*async_result_wrappers):
    """
    Just like :py:func:`as_completed`, but returns only the first item and discards the
    rest.

    :param async_result_wrappers:
    :return:

    .. versionadded:: 0.5.0
    """
    for item in async_result_wrappers:
        if not isinstance(item, AsyncMethodCall):
            raise TypeError("Got non-AsyncMethodCall object: {}".format(item))
    wrappers_copy = list(async_result_wrappers)
    while True:
        completed = list(filter(lambda x: x.finished(), wrappers_copy))
        if not len(completed):
            continue

        return completed[0].result()