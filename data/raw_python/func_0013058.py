def _find_methods(cls, *names, **kwds):
    """Compute a list of composable methods.

    Because this is a common operation and the class hierarchy is
    static, the outcome is cached (assuming that for a particular list
    of names the reversed flag is either always on, or always off).

    Args:
      *names: One or more method names.
      reverse: Optional flag, default False; if True, the list is
        reversed.

    Returns:
      A list of callable class method objects.
    """
    reverse = kwds.pop('reverse', False)
    assert not kwds, repr(kwds)
    cache = cls.__dict__.get('_find_methods_cache')
    if cache:
      hit = cache.get(names)
      if hit is not None:
        return hit
    else:
      cls._find_methods_cache = cache = {}
    methods = []
    for c in cls.__mro__:
      for name in names:
        method = c.__dict__.get(name)
        if method is not None:
          methods.append(method)
    if reverse:
      methods.reverse()
    cache[names] = methods
    return methods