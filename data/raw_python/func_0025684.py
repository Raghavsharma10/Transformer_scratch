def show(*args, **kw):
    """Print value of IRAF or OS environment variables."""

    if len(kw):
        raise TypeError('unexpected keyword argument: %r' % list(kw))

    if args:
        for arg in args:
            print(envget(arg))
    else:
        # print them all
        listVars(prefix="    ", equals="=")