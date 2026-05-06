def grouper(iterable, n):
    '''Collect data into fixed-length chunks or blocks'''
    args = [iter(iterable)] * n
    for group in izip_longest(fillvalue=None, *args):
        group = [g for g in group if g != None]
        yield group