def category(**kwargs):
    """Get a category."""
    if 'series' in kwargs:
        kwargs.pop('series')
        path = 'series'
    else:
        path = None
    return Fred().category(path, **kwargs)