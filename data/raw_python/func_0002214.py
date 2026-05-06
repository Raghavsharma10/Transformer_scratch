def read_var_int(file_obj):
    """Read a variable-length integer.

    Parameters
    ----------
    file_obj : file-like object
        The file to read from.

    Returns
    -------
    int
        the variable-length value read

    """
    # Read all bytes from here, stopping with the first one that does not have
    # the MSB set. Save the lower 7 bits, and keep stacking to the *left*.
    val = 0
    shift = 0
    while True:
        # Read next byte
        next_val = ord(file_obj.read(1))
        val |= ((next_val & 0x7F) << shift)
        shift += 7
        if not next_val & 0x80:
            break

    return val