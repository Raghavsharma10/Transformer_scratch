def calling_code(f, f_name=None, raise_for_missing=True):
    """Return the code string for calling a function. """
    import inspect
    from ambry.dbexceptions import ConfigurationError

    if inspect.isclass(f):
        try:
            args = inspect.getargspec(f.__init__).args
        except TypeError as e:
            raise TypeError("Failed to inspect {}: {}".format(f, e))

    else:
        args = inspect.getargspec(f).args

    if len(args) > 1 and args[0] == 'self':
        args = args[1:]

    for a in args:
        if a not in all_args + ('exception',):  # exception arg is only for exception handlers
            if raise_for_missing:
                raise ConfigurationError('Caster code {} has unknown argument '
                                         'name: \'{}\'. Must be one of: {} '.format(f, a, ','.join(all_args)))

    arg_map = {e: e for e in var_args}

    args = [arg_map.get(a, a) for a in args]

    return "{}({})".format(f_name if f_name else f.__name__, ','.join(args))