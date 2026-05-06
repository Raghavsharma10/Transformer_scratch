def convert_bytes(value):
    """
    Reduces bytes to more convenient units (i.e. KiB, GiB, TiB, etc.).

    Args:
        values (int): Value in Bytes

    Returns:
        tup (tuple): Tuple of value, unit (e.g. (10, 'MiB'))
    """
    n = np.rint(len(str(value))/4).astype(int)
    return value/(1024**n), sizes[n]