def maybe_obj(str_or_obj):
    """If argument is not a string, return it.

    Otherwise import the dotted name and return that.
    """
    if not isinstance(str_or_obj, six.string_types):
        return str_or_obj
    parts = str_or_obj.split(".")
    mod, modname = None, None
    for p in parts:
        modname = p if modname is None else "%s.%s" % (modname, p)
        try:
            mod = __import__(modname)
        except ImportError:
            if mod is None:
                raise
            break
    obj = mod
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj