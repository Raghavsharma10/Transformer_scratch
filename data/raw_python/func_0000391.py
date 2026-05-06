def new(expr, *args, **kwargs):
    """
    Template an object.

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
        >>> new_object = uqbar.objects.new(my_object, foo=666, bar=1234)
        >>> print(uqbar.objects.get_repr(new_object))
        MyObject(
            'a',
            'b',
            'c',
            'd',
            bar=1234,
            foo=666,
            quux=['y', 'z'],
            )

    Original object is unchanged:

    ::

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
    # TODO: Clarify old vs. new variable naming here.
    current_args, current_var_args, current_kwargs = get_vars(expr)
    new_kwargs = current_kwargs.copy()

    recursive_arguments = {}
    for key in tuple(kwargs):
        if "__" in key:
            value = kwargs.pop(key)
            key, _, subkey = key.partition("__")
            recursive_arguments.setdefault(key, []).append((subkey, value))

    for key, pairs in recursive_arguments.items():
        recursed_object = current_args.get(key, current_kwargs.get(key))
        if recursed_object is None:
            continue
        kwargs[key] = new(recursed_object, **dict(pairs))

    if args:
        current_var_args = args
    for key, value in kwargs.items():
        if key in current_args:
            current_args[key] = value
        else:
            new_kwargs[key] = value

    new_args = list(current_args.values()) + list(current_var_args)
    return type(expr)(*new_args, **new_kwargs)