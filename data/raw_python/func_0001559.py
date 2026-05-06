def _open(filename, mode="r"):
    """
    Universal open file facility.
    With normal files, this function behaves as the open builtin.
    With gzip-ed files, it decompress or compress according to the specified mode.
    In addition, when filename is '-', it opens the standard input or output according to
    the specified mode.
    Mode are expected to be either 'r' or 'w'.
    """
    if filename.endswith(".gz"):
        return GzipFile(filename, mode, COMPRESSION_LEVEL)
    elif filename == "-":
        if mode == "r":
            return _stdin
        elif mode == "w":
            return _stdout
    else:
        # TODO: set encoding to UTF-8?
        return open(filename, mode=mode)