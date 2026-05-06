def check_buffer(coords, length, buffer):
    """
    check to see how much of the buffer is being used
    """
    s = min(coords[0], buffer)
    e = min(length - coords[1], buffer)
    return [s, e]