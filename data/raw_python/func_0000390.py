def get_vars(expr):
    """
    Get ``args``, ``var args`` and ``kwargs`` for an object ``expr``.

    ::

        >>> class MyObject:
        ...     def __init__(self, arg1, arg2, *var_args, foo=None, bar=None, **kwargs):
        ...         self.arg1 = arg1
        ...         self.arg2 = arg2
        ...         self.var_args = var_args
        ...         self.foo = foo
        ...         self.bar = bar
        ...         self.kwargs = kwargs
        ...
        >>> my_object = MyObject('a', 'b', 'c', 'd', foo='x', quux=['y', 'z'])

    ::

        >>> import uqbar
        >>> args, var_args, kwargs = uqbar.objects.get_vars(my_object)

    ::

        >>> args
        OrderedDict([('arg1', 'a'), ('arg2', 'b')])

    ::

        >>> var_args
        ['c', 'd']

    ::

        >>> kwargs
        {'foo': 'x', 'quux': ['y', 'z']}

    """
    # print('TYPE?', type(expr))
    signature = _get_object_signature(expr)
    if signature is None:
        return ({}, [], {})
    # print('SIG?', signature)
    args = collections.OrderedDict()
    var_args = []
    kwargs = {}
    if expr is None:
        return args, var_args, kwargs
    for i, (name, parameter) in enumerate(signature.parameters.items()):
        # print('   ', parameter)

        if i == 0 and name in ("self", "cls", "class_", "klass"):
            continue

        if parameter.kind is inspect._POSITIONAL_ONLY:
            try:
                args[name] = getattr(expr, name)
            except AttributeError:
                args[name] = expr[name]

        elif (
            parameter.kind is inspect._POSITIONAL_OR_KEYWORD
            or parameter.kind is inspect._KEYWORD_ONLY
        ):
            found = False
            for x in (name, "_" + name):
                try:
                    value = getattr(expr, x)
                    found = True
                    break
                except AttributeError:
                    try:
                        value = expr[x]
                        found = True
                        break
                    except (KeyError, TypeError):
                        pass
            if not found:
                raise ValueError("Cannot find value for {!r}".format(name))
            if parameter.default is inspect._empty:
                args[name] = value
            elif parameter.default != value:
                kwargs[name] = value

        elif parameter.kind is inspect._VAR_POSITIONAL:
            value = None
            try:
                value = expr[:]
            except TypeError:
                value = getattr(expr, name)
            if value:
                var_args.extend(value)

        elif parameter.kind is inspect._VAR_KEYWORD:
            items = {}
            if hasattr(expr, "items"):
                items = expr.items()
            elif hasattr(expr, name):
                mapping = getattr(expr, name)
                if not isinstance(mapping, dict):
                    mapping = dict(mapping)
                items = mapping.items()
            elif hasattr(expr, "_" + name):
                mapping = getattr(expr, "_" + name)
                if not isinstance(mapping, dict):
                    mapping = dict(mapping)
                items = mapping.items()
            for key, value in items:
                if key not in args:
                    kwargs[key] = value

    return args, var_args, kwargs