def releases(release_id=None, **kwargs):
    """Get all releases of economic data."""
    if not 'id' in kwargs and release_id is not None:
        kwargs['release_id'] = release_id
        return Fred().release(**kwargs)
    return Fred().releases(**kwargs)