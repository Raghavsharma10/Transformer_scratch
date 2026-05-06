def print_doc1(*args, **kwargs):
    '''Print the first paragraph of the docstring of the decorated function.

    The paragraph will be printed as a oneliner.

    May be invoked as a simple, argument-less decorator (i.e. ``@print_doc1``)
    or with named arguments ``color``, ``bold``, ``prefix`` of ``tail``
    (eg. ``@print_doc1(color=utils.red, bold=True, prefix=' ')``).

    Examples:
    #    >>> @print_doc1
    #    ... def foo():
    #    ...     """First line of docstring.
    #    ...
    #    ...     another line.
    #    ...     """
    #    ...     pass
    #    ...
    #    >>> foo()
    #    \033[34mFirst line of docstring\033[0m

    #    >>> @print_doc1
    #    ... def foo():
    #    ...     """First paragraph of docstring which contains more than one
    #    ...     line.
    #    ...
    #    ...     Another paragraph.
    #    ...     """
    #    ...     pass
    #    ...
    #    >>> foo()
    #    \033[34mFirst paragraph of docstring which contains more than one line\033[0m
    '''
    # output settings from kwargs or take defaults
    color = kwargs.get('color', blue)
    bold = kwargs.get('bold', False)
    prefix = kwargs.get('prefix', '')
    tail = kwargs.get('tail', '\n')

    def real_decorator(func):
        '''real decorator function'''
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''the wrapper function'''
            try:
                prgf = first_paragraph(func.__doc__)
                print(color(prefix + prgf + tail, bold))
            except AttributeError as exc:
                name = func.__name__
                print(red(flo('{name}() has no docstring')))
                raise(exc)
            return func(*args, **kwargs)
        return wrapper

    invoked = bool(not args or kwargs)
    if not invoked:
        # invoke decorator function which returns the wrapper function
        return real_decorator(func=args[0])

    return real_decorator