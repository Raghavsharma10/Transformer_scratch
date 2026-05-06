def object_from_string(name):
    """Creates a Python class or function from its fully qualified name.

    :param name: A fully qualified name of a class or a function. In Python 3 this
        is only allowed to be of text type (unicode). In Python 2, both bytes and unicode
        are allowed.
    :return: A function or class object.

    This method is used by serialization code to create a function or class
    from a fully qualified name.

    """
    if six.PY3:
        if not isinstance(name, str):
            raise TypeError("name must be str, not %r" % type(name))
    else:
        if isinstance(name, unicode):
            name = name.encode("ascii")
        if not isinstance(name, (str, unicode)):
            raise TypeError("name must be bytes or unicode, got %r" % type(name))

    pos = name.rfind(".")
    if pos < 0:
        raise ValueError("Invalid function or class name %s" % name)
    module_name = name[:pos]
    func_name = name[pos + 1 :]
    try:
        mod = __import__(module_name, fromlist=[func_name], level=0)
    except ImportError:
        # Hail mary. if the from import doesn't work, then just import the top level module
        # and do getattr on it, one level at a time. This will handle cases where imports are
        # done like `from . import submodule as another_name`
        parts = name.split(".")
        mod = __import__(parts[0], level=0)
        for i in range(1, len(parts)):
            mod = getattr(mod, parts[i])
        return mod

    else:
        return getattr(mod, func_name)