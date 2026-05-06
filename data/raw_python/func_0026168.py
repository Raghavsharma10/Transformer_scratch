def flatten_errors(cfg, res, levels=None, results=None):
    """
    An example function that will turn a nested dictionary of results
    (as returned by ``ConfigObj.validate``) into a flat list.

    ``cfg`` is the ConfigObj instance being checked, ``res`` is the results
    dictionary returned by ``validate``.

    (This is a recursive function, so you shouldn't use the ``levels`` or
    ``results`` arguments - they are used by the function.)

    Returns a list of keys that failed. Each member of the list is a tuple::

        ([list of sections...], key, result)

    If ``validate`` was called with ``preserve_errors=False`` (the default)
    then ``result`` will always be ``False``.

    *list of sections* is a flattened list of sections that the key was found
    in.

    If the section was missing (or a section was expected and a scalar provided
    - or vice-versa) then key will be ``None``.

    If the value (or section) was missing then ``result`` will be ``False``.

    If ``validate`` was called with ``preserve_errors=True`` and a value
    was present, but failed the check, then ``result`` will be the exception
    object returned. You can use this as a string that describes the failure.

    For example *The value "3" is of the wrong type*.
    """
    if levels is None:
        # first time called
        levels = []
        results = []
    if res == True:
        return results
    if res == False or isinstance(res, Exception):
        results.append((levels[:], None, res))
        if levels:
            levels.pop()
        return results
    for (key, val) in res.items():
        if val == True:
            continue
        if isinstance(cfg.get(key), dict):
            # Go down one level
            levels.append(key)
            flatten_errors(cfg[key], val, levels, results)
            continue
        results.append((levels[:], key, val))
    #
    # Go up one level
    if levels:
        levels.pop()
    #
    return results