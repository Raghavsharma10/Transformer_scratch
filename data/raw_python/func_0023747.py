def assignIfExists(opts, default=None, **kwargs):
    """
    Helper for assigning object attributes from API responses.
    """
    for opt in opts:
        if(opt in kwargs):
            return kwargs[opt]
    return default