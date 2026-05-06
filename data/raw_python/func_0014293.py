def assertinstance(obj, types):
    """
    Make sure object `obj` is of type `types`. Else, raise TypeError.
    """
    if isinstance(obj, types):
        return obj
    raise TypeError('{} must be instance of {}'.format(obj, types))