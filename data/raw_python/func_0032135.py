def ints_to_string(ints):
    """Convert a list of integers to a *|* separated string.

    Args:
        ints (list[int]|int): List of integer items to convert or single
            integer to convert.

    Returns:
        str: Formatted string
    """
    if not isinstance(ints, list):
        return six.u(str(ints))

    return '|'.join(six.u(str(l)) for l in ints)