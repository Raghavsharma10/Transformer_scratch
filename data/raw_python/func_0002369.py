def series(identifier=None, **kwargs):
    """Get an economic data series."""
    if identifier:
        kwargs['series_id'] = identifier
    if 'release' in kwargs:
        kwargs.pop('release')
        path = 'release'
    elif 'releases' in kwargs:
        kwargs.pop('releases')
        path = 'release'
    else:
        path = None
    return Fred().series(path, **kwargs)