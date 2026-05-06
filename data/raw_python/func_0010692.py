def get_methods(*objs):
    """ Return the names of all callable attributes of an object"""
    return set(
        attr
        for obj in objs
        for attr in dir(obj)
        if not attr.startswith('_') and callable(getattr(obj, attr))
    )