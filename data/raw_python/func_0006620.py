def _unichr(i):
    """
    Helper function for taking a Unicode scalar value and returning a Unicode character.

    :param s: Unicode scalar value to convert.
    :return: Unicode character
    """
    if not isinstance(i, int):
        raise TypeError
    try:
        return six.unichr(i)
    except ValueError:
        # Workaround the error "ValueError: unichr() arg not in range(0x10000) (narrow Python build)"
        return struct.pack("i", i).decode("utf-32")