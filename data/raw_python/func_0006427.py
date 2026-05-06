def zip_nofill(*iterables):
    '''Zipping iterables without fillvalue.

    Note: https://stackoverflow.com/questions/38054593/zip-longest-without-fillvalue
    '''
    return (tuple([entry for entry in iterable if entry is not None]) for iterable in itertools.izip_longest(*iterables, fillvalue=None))