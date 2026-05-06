def fill_padding(padded_string):
    # type: (bytes) -> bytes
    """
    Fill up missing padding in a string.

    This function makes sure that the string has length which is multiplication of 4,
    and if not, fills the missing places with dots.

    :param str padded_string: string to be decoded that might miss padding dots.
    :return: properly padded string
    :rtype: str
    """
    length = len(padded_string)
    reminder = len(padded_string) % 4
    if reminder:
        return padded_string.ljust(length + 4 - reminder, b'.')
    return padded_string