def throws(exc):
    '''
    throws(exc)(func) -> func'

    Function decorator. Used to decorate a function that raises exc.
    '''
    if not __CHECKING__:
        return lambda f: f

    def wrap(f):
        def call(*args, **kwd):
            res = f(*args, **kwd)
            # raise UncheckedExceptionError if exc is not automatically
            # registered by a function decorated with @catches(exc);
            # otherwise do nothing
            exc_checker.throwing(exc)
            return res
        call.__name__ = f.__name__
        return call
    return wrap