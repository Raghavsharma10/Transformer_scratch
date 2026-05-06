def dot_path(obj: t.Union[t.Dict, object],
             path: str,
             default: t.Any = None,
             separator: str = '.'):
    """
    Provides an access to elements of a mixed dict/object type by a delimiter-separated path.
    ::

        class O1:
            my_dict = {'a': {'b': 1}}

        class O2:
            def __init__(self):
                self.nested = O1()

        class O3:
            final = O2()

        o = O3()
        assert utils.dot_path(o, 'final.nested.my_dict.a.b') == 1

    .. testoutput::

        True

    :param obj: object or dict
    :param path: path to value
    :param default: default value if chain resolve failed
    :param separator: ``.`` by default
    :return: value or default
    """
    path_items = path.split(separator)
    val = obj
    sentinel = object()
    for item in path_items:
        if isinstance(val, dict):
            val = val.get(item, sentinel)
            if val is sentinel:
                return default
        else:
            val = getattr(val, item, sentinel)
            if val is sentinel:
                return default
    return val