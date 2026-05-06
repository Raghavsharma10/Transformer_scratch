def dict_copy(func):
    "copy dict args, to avoid modifying caller's copy"
    def proxy(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        for var in kwargs:
            if isinstance(kwargs[var], dict):
                new_kwargs[var] = dict(kwargs[var])
            else:
                new_kwargs[var] = kwargs[var]
        for arg in args:
            if isinstance(arg, dict):
                new_args.append(dict(arg))
            else:
                new_args.append(arg)
        return func(*new_args, **new_kwargs)
    return proxy