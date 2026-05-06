def _padded_hex(i, pad_width=4, uppercase=True):
    """
    Helper function for taking an integer and returning a hex string.  The string will be padded on the left with zeroes
    until the string is of the specified width.  For example:

    _padded_hex(31, pad_width=4, uppercase=True) -> "001F"

    :param i: integer to convert to a hex string
    :param pad_width: (int specifying the minimum width of the output string.  String will be padded on the left with '0'
                      as needed.
    :param uppercase: Boolean indicating if we should use uppercase characters in the output string (default=True).
    :return: Hex string representation of the input integer.
    """
    result = hex(i)[2:]  # Remove the leading "0x"
    if uppercase:
        result = result.upper()
    return result.zfill(pad_width)