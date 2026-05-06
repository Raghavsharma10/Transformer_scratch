def _replace_star(fmt, size):
    """
    Replace the `*` placeholder in a format string (fmt), so that
    struct.calcsize(fmt) is equal to the given `size` using the format
    following the placeholder.

    Raises `ValueError` if number of `*` is larger than 1. If no `*`
    in `fmt`, returns `fmt` without checking its size!

    Examples
    --------
    >>> _replace_star('ii*fi', 40)
    'ii7fi'
    """
    n_stars = fmt.count('*')

    if n_stars > 1:
        raise ValueError("More than one `*` in format (%s)." % fmt)

    if n_stars:
        i = fmt.find('*')
        s = struct.calcsize(fmt.replace(fmt[i:i + 2], ''))
        n = old_div((size - s), struct.calcsize(fmt[i + 1]))

        fmt = fmt.replace('*', str(n))

    return fmt