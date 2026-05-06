def print_full_name(*args, **kwargs):
    '''Decorator, print the full name of the decorated function.

    May be invoked as a simple, argument-less decorator (i.e. ``@print_doc1``)
    or with named arguments ``color``, ``bold``, or ``prefix``
    (eg. ``@print_doc1(color=utils.red, bold=True, prefix=' ')``).
    '''
    color = kwargs.get('color', default_color)
    bold = kwargs.get('bold', False)
    prefix = kwargs.get('prefix', '')
    tail = kwargs.get('tail', '')

    def real_decorator(func):
        '''real decorator function'''
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''the wrapper function'''
            first_line = ''
            try:
                first_line = func.__module__ + '.' + func.__qualname__
            except AttributeError as exc:
                first_line = func.__name__
            print(color(prefix + first_line + tail, bold))
            return func(*args, **kwargs)
        return wrapper

    invoked = bool(not args or kwargs)
    if not invoked:
        # invoke decorator function which returns the wrapper function
        return real_decorator(func=args[0])

    return real_decorator