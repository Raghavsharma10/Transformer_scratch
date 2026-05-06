def better_ts_function(f):
    '''Decorator which check if timeseries has a better
 implementation of the function.'''
    fname = f.__name__

    def _(ts, *args, **kwargs):
        func = getattr(ts, fname, None)
        if func:
            return func(*args, **kwargs)
        else:
            return f(ts, *args, **kwargs)

    _.__name__ = fname

    return _