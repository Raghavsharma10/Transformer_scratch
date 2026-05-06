def pretty_memory_info():
    '''
    Pretty format memory info.

    Returns
    -------
    str
        Memory info.

    Examples
    --------
    >>> pretty_memory_info()
    '5MB memory usage'
    '''
    process = psutil.Process(os.getpid())
    return '{}MB memory usage'.format(int(process.memory_info().rss / 2**20))