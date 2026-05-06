def sources(source_id=None, **kwargs):
    """Get the sources of economic data."""
    if source_id or 'id' in kwargs:
        return source(source_id, **kwargs)
    return Fred().sources(**kwargs)