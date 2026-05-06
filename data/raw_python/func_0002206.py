def read_block(fobj):
    """Read a block.

    Reads a block from a file object by first reading the number of bytes to read, which must
    be encoded as a variable-byte length integer.

    Parameters
    ----------
    fobj : file-like object
        The file to read from.

    Returns
    -------
    bytes
        block of bytes read

    """
    num = read_var_int(fobj)
    log.debug('Next block: %d bytes', num)
    return fobj.read(num)