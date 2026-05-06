def retry(*dargs, **dkwargs):
    """
    The retry decorator. Can be passed all the arguments that are accepted by
    Retry.__init__.
    """
    # support both @retry and @retry() as valid syntax
    if len(dargs) == 1 and callable(dargs[0]):
        def wrap_simple(f):

            @_wraps(f)
            def wrapped_f(*args, **kw):
                return Retry()(f, *args, **kw)

            return wrapped_f

        return wrap_simple(dargs[0])

    else:
        def wrap(f):

            @_wraps(f)
            def wrapped_f(*args, **kw):
                return Retry(*dargs, **dkwargs)(f, *args, **kw)

            return wrapped_f

        return wrap