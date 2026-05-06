def ISINSTANCE(instance, A_tuple):  # noqa
    """
    Allows you to do isinstance checks on futures.
    Really, I discourage this because duck-typing is usually better.
    But this can provide you with a way to use isinstance with futures.
    Works with other objects too.

    :param instance:
    :param A_tuple:
    :return:
    """
    try:
        instance = instance._redpipe_future_result
    except AttributeError:
        pass

    return isinstance(instance, A_tuple)