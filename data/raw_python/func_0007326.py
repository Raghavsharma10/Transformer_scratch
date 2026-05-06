def _get_base(obj):
    """Unwrap decorators to retrieve the base object."""
    if hasattr(obj, '__func__'):
        obj = obj.__func__
    elif isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, (classmethod, staticmethod)):
        # Fallback for Python < 2.7 back when no `__func__` attribute
        # was defined for those descriptors.
        obj = obj.__get__(None, object)
    else:
        return obj

    return _get_base(obj)