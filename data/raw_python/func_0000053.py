def get_obj(path):
    """Return obj for given dotted path.

    Typical inputs for `path` are 'os' or 'os.path' in which case you get a
    module; or 'os.path.exists' in which case you get a function from that
    module.

    Just returns the given input in case it is not a str.

    Note: Relative imports not supported.
    Raises ImportError or AttributeError as appropriate.

    """
    # Since we usually pass in mocks here; duck typing is not appropriate
    # (mocks respond to every attribute).
    if not isinstance(path, str):
        return path

    if path.startswith('.'):
        raise TypeError('relative imports are not supported')

    parts = path.split('.')
    head, tail = parts[0], parts[1:]

    obj = importlib.import_module(head)

    # Normally a simple reduce, but we go the extra mile
    # for good exception messages.
    for i, name in enumerate(tail):
        try:
            obj = getattr(obj, name)
        except AttributeError:
            # Note the [:i] instead of [:i+1], so we get the path just
            # *before* the AttributeError, t.i. the part of it that went ok.
            module = '.'.join([head] + tail[:i])
            try:
                importlib.import_module(module)
            except ImportError:
                raise AttributeError(
                    "object '%s' has no attribute '%s'" % (module, name))
            else:
                raise AttributeError(
                    "module '%s' has no attribute '%s'" % (module, name))
    return obj