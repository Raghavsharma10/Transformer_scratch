def get_repr(expr, multiline=False):
    """
    Build a repr string for ``expr`` from its vars and signature.

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
        >>> print(uqbar.objects.get_repr(my_object))
        MyObject(
            'a',
            'b',
            'c',
            'd',
            foo='x',
            quux=['y', 'z'],
            )

    """
    signature = _get_object_signature(expr)
    if signature is None:
        return "{}()".format(type(expr).__name__)

    defaults = {}
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect._empty:
            defaults[name] = parameter.default

    args, var_args, kwargs = get_vars(expr)
    args_parts = collections.OrderedDict()
    var_args_parts = []
    kwargs_parts = {}
    has_lines = multiline
    parts = []

    # Format keyword-optional arguments.
    # print(type(expr), args)
    for i, (key, value) in enumerate(args.items()):
        arg_repr = _dispatch_formatting(value)
        if "\n" in arg_repr:
            has_lines = True
        args_parts[key] = arg_repr

    # Format *args
    for arg in var_args:
        arg_repr = _dispatch_formatting(arg)
        if "\n" in arg_repr:
            has_lines = True
        var_args_parts.append(arg_repr)

    # Format **kwargs
    for key, value in sorted(kwargs.items()):
        if key in defaults and value == defaults[key]:
            continue
        value = _dispatch_formatting(value)
        arg_repr = "{}={}".format(key, value)
        has_lines = True
        kwargs_parts[key] = arg_repr

    for _, part in args_parts.items():
        parts.append(part)
    parts.extend(var_args_parts)
    for _, part in sorted(kwargs_parts.items()):
        parts.append(part)

    # If we should format on multiple lines, add the appropriate formatting.
    if has_lines and parts:
        for i, part in enumerate(parts):
            parts[i] = "\n".join("    " + line for line in part.split("\n"))
        parts.append("    )")
        parts = ",\n".join(parts)
        return "{}(\n{}".format(type(expr).__name__, parts)

    parts = ", ".join(parts)
    return "{}({})".format(type(expr).__name__, parts)