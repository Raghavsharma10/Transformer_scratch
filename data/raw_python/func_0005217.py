def unwrap_obj(obj):
    """ Gets the actual object from a decorated or wrapped function
        @obj: (#object) the object to unwrap
    """
    try:
        obj = obj.fget
    except (AttributeError, TypeError):
        pass
    try:
        # Cached properties
        if obj.func.__doc__ == obj.__doc__:
            obj = obj.func
    except AttributeError:
        pass
    try:
        # Setter/Getters
        obj = obj.getter
    except AttributeError:
        pass
    try:
        # Wrapped Funcs
        obj = inspect.unwrap(obj)
    except:
        pass
    return obj