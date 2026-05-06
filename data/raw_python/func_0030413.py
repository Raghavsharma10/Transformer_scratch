def trace(fn): # pragma: no cover
    """ Prints parameteters and return values of the each call of the wrapped function.

    Usage:
        decorate appropriate function or method:
            @trace
            def myf():
                ...
    """
    def wrapped(*args, **kwargs):
        msg = []
        msg.append('Enter {}('.format(fn.__name__))

        if args:
            msg.append(', '.join([str(x) for x in args]))

        if kwargs:
            kwargs_str = ', '.join(['{}={}'.format(k, v) for k, v in list(kwargs.items())])
            if args:
                msg.append(', ')
            msg.append(kwargs_str)
        msg.append(')')
        print(''.join(msg))
        ret = fn(*args, **kwargs)
        print('Return {}'.format(ret))
        return ret
    return wrapped