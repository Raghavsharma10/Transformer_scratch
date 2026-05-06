def IS(instance, other):  # noqa
    """
    Support the `future is other` use-case.
    Can't override the language so we built a function.
    Will work on non-future objects too.

    :param instance: future or any python object
    :param other: object to compare.
    :return:
    """
    try:
        instance = instance._redpipe_future_result  # noqa
    except AttributeError:
        pass

    try:
        other = other._redpipe_future_result
    except AttributeError:
        pass

    return instance is other