def filter_response_size(size):
    """Filter :class:`.Line` objects by the response size (in bytes).

    Specially useful when looking for big file downloads.

    :param size: Minimum amount of bytes a response body weighted.
    :type size: string
    :returns: a function that filters by the response size.
    :rtype: function
    """
    if size.startswith('+'):
        size_value = int(size[1:])
    else:
        size_value = int(size)

    def filter_func(log_line):
        bytes_read = log_line.bytes_read
        if bytes_read.startswith('+'):
            bytes_read = int(bytes_read[1:])
        else:
            bytes_read = int(bytes_read)

        return bytes_read >= size_value

    return filter_func