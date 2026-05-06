def human_size(size):
    """ Return a human-readable representation of a byte size.

        @param size: Number of bytes as an integer or string.
        @return: String of length 10 with the formatted result.
    """
    if isinstance(size, string_types):
        size = int(size, 10)

    if size < 0:
        return "-??? bytes"

    if size < 1024:
        return "%4d bytes" % size
    for unit in ("KiB", "MiB", "GiB"):
        size /= 1024.0
        if size < 1024:
            return "%6.1f %s" % (size, unit)

    return "%6.1f GiB" % size