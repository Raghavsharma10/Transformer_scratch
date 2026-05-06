def source(source_id=None, **kwargs):
    """Get a source of economic data."""
    if source_id is not None:
        kwargs['source_id'] = source_id
    elif 'id' in kwargs:
        source_id = kwargs.pop('id')
        kwargs['source_id'] = source_id
    if 'releases' in kwargs:
        kwargs.pop('releases')
        path = 'releases'
    else:
        path = None
    return Fred().source(path, **kwargs)