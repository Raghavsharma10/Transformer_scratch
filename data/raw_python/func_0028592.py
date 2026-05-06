def _groups_of_size(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # _groups_of_size('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)