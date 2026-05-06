def _length_hint(obj):
    """Returns the length hint of an object."""
    try:
        return len(obj)
    except TypeError:
        try:
            get_hint = type(obj).__length_hint__
        except AttributeError:
            return None
        try:
            hint = get_hint(obj)
        except TypeError:
            return None
        if hint is NotImplemented or \
           not isinstance(hint, (int, long)) or \
           hint < 0:
            return None
        return hint